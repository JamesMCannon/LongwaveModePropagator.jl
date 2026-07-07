#==
Excitation factor, height gain functions, and electric field mode sum
==#

# Rows of the field matrix returned by `fieldsum`: (Ez, Ey, Ex, Z₀Hz, Z₀Hy, Z₀Hx).
# Magnetic rows carry Z₀H so all six rows share units of V/m and the same Q scaling;
# see `modeterms`.
const NUMFIELDCOMPONENTS = 6

"""
    ExcitationFactor{T,T2}

Constants used in calculating excitation factors and height gains.

# Fields

- `F₁::T`: height gain constant. See [Pappert1976].
- `F₂::T`
- `F₃::T`
- `F₄::T`
- `h₁0::T`: first modified Hankel function of order 1/3 at the ground.
- `h₂0::T`: second modified Hankel function of order 1/3 at the ground.
- `EyHy::T`: polarization ratio ``Ey/Hy``, derived from reflection coefficients (or ``T``s).
- `Rg::T2`: ground reflection coefficient matrix.

# References

[Pappert1976]: R. A. Pappert and L. R. Shockey, “Simplified VLF/LF mode conversion program
    with allowance for elevated, arbitrarily oriented electric dipole antennas,” Naval
    Electronics Laboratory Center, San Diego, CA, Interim Report 771, Oct. 1976. [Online].
    Available: http://archive.org/details/DTIC_ADA033412.
"""
struct ExcitationFactor{T,T2}
    F₁::T
    F₂::T
    F₃::T
    F₄::T
    h₁0::T
    h₂0::T
    EyHy::T
    Rg::T2
end

"""
    excitationfactorconstants(ea₀, R, Rg, frequency, ground; params=LMPParams())

Return an `ExcitationFactor` struct used in calculating height-gain functions and excitation
factors where eigenangle `ea₀` is referenced to the ground.

!!! note

    This function assumes that reflection coefficients `R` and `Rg` are referenced to
    ``d = z = 0``.

# References

[Pappert1976]: R. A. Pappert and L. R. Shockey, “Simplified VLF/LF mode conversion program
    with allowance for elevated, arbitrarily oriented electric dipole antennas,” Naval
    Electronics Laboratory Center, San Diego, CA, Interim Report 771, Oct. 1976. [Online].
    Available: http://archive.org/details/DTIC_ADA033412.

[Ferguson1980]: J. A. Ferguson and F. P. Snyder, “Approximate VLF/LF waveguide mode
    conversion model: Computer applications: FASTMC and BUMP,” Naval Ocean Systems Center,
    San Diego, CA, NOSC-TD-400, Nov. 1980. [Online].
    Available: http://www.dtic.mil/docs/citations/ADA096240.

[Morfitt1980]: D. G. Morfitt, “‘Simplified’ VLF/LF mode conversion computer programs:
    GRNDMC and ARBNMC,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-514, Jan. 1980.
    [Online]. Available: http://www.dtic.mil/docs/citations/ADA082695.
"""
function excitationfactorconstants(ea₀, R, Rg, frequency, ground; params=LMPParams())
    S², C² = ea₀.sin²θ, ea₀.cos²θ
    k, ω = frequency.k, frequency.ω
    ϵᵣ, σ = ground.ϵᵣ, ground.σ

    @unpack earthradius = params

    # Precompute
    α = 2/earthradius
    tmp1 = pow23(α/k)/2  # 1/2*(a/k)^(2/3)

    q₀ = pow23(k/α)*C²  # (a/k)^(-2/3)*C²
    h₁0, h₂0, dh₁0, dh₂0 = modifiedhankel(q₀)

    H₁0 = dh₁0 + tmp1*h₁0
    H₂0 = dh₂0 + tmp1*h₂0

    n₀² = 1  # modified free space index of refraction squared, referenced to ground
    Ng² = complex(ϵᵣ, -σ/(ω*E0))  # ground index of refraction

    # Precompute
    tmp2 = 1im*cbrt(k/α)*sqrt(Ng² - S²)  # i(k/α)^(1/3)*(Ng² - S²)^(1/2)

    F₁ = -H₂0 + (n₀²/Ng²)*tmp2*h₂0
    F₂ = H₁0 - (n₀²/Ng²)*tmp2*h₁0
    F₃ = -dh₂0 + tmp2*h₂0
    F₄ = dh₁0 - tmp2*h₁0

    # ``EyHy = ey/hy``. Also known as `f0fr` or `f`.
    # It is a polarization ratio that adds the proper amount of TE wave when the y component
    # of the magnetic field is normalized to unity at the ground.
    # A principally TE mode will have `1 - R[1,1]*Rg[1,1]` very small and EyHy will be very
    # small, so we use the first equation below. Conversely, a principally TM mode will have
    # `1 - R[2,2]Rg[2,2]` very small and EyHy very large, resulting in the use of the second
    # equation below. [Ferguson1980] pg. 58 seems to suggest the use of the opposite, but
    # LWPC uses the form used here and this makes sense because there are more working
    # decimal places.
    if abs2(1 - R[1,1]*Rg[1,1]) < abs2(1 - R[2,2]*Rg[2,2])
        # EyHy = T₃/T₁
        EyHy = (1 + Rg[2,2])*R[2,1]*Rg[1,1]/((1 + Rg[1,1])*(1 - R[2,2]*Rg[2,2]))
    else
        # EyHy = T₂/(T₃*T₄)
        EyHy = (1 + Rg[2,2])*(1 - R[1,1]*Rg[1,1])/((1 + Rg[1,1])*R[1,2]*Rg[2,2])
    end

    return ExcitationFactor(F₁, F₂, F₃, F₄, h₁0, h₂0, EyHy, Rg)
end

"""
    excitationfactor(ea, dFdθ, R, Rg, efconstants::ExcitationFactor; params=LMPParams())

Compute excitation factors for the ``Hy`` field at the emitter returned as the tuple
`(λv, λb, λe)` for vertical, broadside, and end-on dipoles. `dFdθ` is the derivative of the
modal equation with respect to ``θ``.

The excitation factor describes how efficiently the field component can be excited in the
waveguide.

This function is similar to the approach taken in [Pappert1983], which makes
use of ``T`` (different from `TMatrix`) rather than ``τ``. From the total ``Hy`` excitation
factor (the sum product of the `λ`s with the antenna orientation terms), the excitation
factor for electric fields can be found as:

- ``λz = -S₀λ``
- ``λx = EyHy⋅λ``
- ``λy = -λ``

!!! note

    This function assumes that reflection coefficients `R` and `Rg` are referenced to
    ``d = z = 0``.

# References

[Morfitt1980]: D. G. Morfitt, “‘Simplified’ VLF/LF mode conversion computer programs:
    GRNDMC and ARBNMC,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-514, Jan. 1980.
    [Online]. Available: http://www.dtic.mil/docs/citations/ADA082695.

[Pappert1983]: R. A. Pappert, L. R. Hitney, and J. A. Ferguson, “ELF/VLF (Extremely Low
    Frequency/Very Low Frequency) long path pulse program for antennas of arbitrary
    elevation and orientation,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-891,
    Aug. 1983. [Online]. Available: http://www.dtic.mil/docs/citations/ADA133876.

[Pappert1986]: R. A. Pappert and J. A. Ferguson, “VLF/LF mode conversion model calculations
    for air to air transmissions in the earth-ionosphere waveguide,” Radio Sci., vol. 21,
    no. 4, pp. 551–558, Jul. 1986, doi: 10.1029/RS021i004p00551.
"""
function excitationfactor(ea, dFdθ, R, efconstants::ExcitationFactor; params=LMPParams())
    S = ea.sinθ
    sqrtS = sqrt(S)
    S₀ = referencetoground(ea.sinθ; params=params)

    @unpack F₁, F₂, F₃, F₄, h₁0, h₂0, Rg = efconstants

    # Unlike the formulations shown in the references, we scale these excitation factors
    # with `D##` instead of `EyHy` and appropriately don't scale the height gains.
    F₁h₁0 = F₁*h₁0
    F₂h₂0 = F₂*h₂0
    F₃h₁0 = F₃*h₁0
    F₄h₂0 = F₄*h₂0

    D₁₁ = (F₁h₁0 + F₂h₂0)^2
    D₁₂ = (F₁h₁0 + F₂h₂0)*(F₃h₁0 + F₄h₂0)
    # D₂₂ = (F₃h₁0 + F₄h₂0)^2

    # `sqrtS` should be at `curvatureheight` because that is where `dFdθ` is evaluated
    T₁ = sqrtS*(1 + Rg[1,1])^2*(1 - R[2,2]*Rg[2,2])/(dFdθ*Rg[1,1]*D₁₁)
    # T₂ = sqrtS*(1 + Rg[2,2])^2*(1 - R[1,1]*Rg[1,1])/(dFdθ*Rg[2,2]*D₂₂)
    T₃ = sqrtS*(1 + Rg[1,1])*(1 + Rg[2,2])*R[2,1]/(dFdθ*D₁₂)
    T₄ = R[1,2]/R[2,1]

    # These are [Pappert1983] terms divided by `-S`, the factor between Hy and Ez
    λv = -S₀*T₁
    λb = T₃*T₄
    λe = T₁

    return λv, λb, λe
end

@doc raw"""
    heightgains(z, ea₀, frequency, efconstants::ExcitationFactor; params=LMPParams())

Compute height gain functions at height `z` returned as the tuple `(fz, fy, fx, dfy)` where
eigenangle `ea₀` is referenced to the ground.

- `fz` is the height gain for the vertical electric field component ``Ez``.
- `fy` is the height gain for the transverse electric field component ``Ey``.
- `fx` is the height gain for the horizontal electric field component ``Ex``.
- `dfy` is the height derivative ``d(fy)/dz`` of the ``Ey`` height gain, used for the
    horizontal magnetic field component ``Hx``.
[Pappert1983]

!!! note

    This function assumes that reflection coefficients are referenced to ``d = z = 0``.

See also: [`excitationfactorconstants`](@ref)

# References

[Pappert1983]: R. A. Pappert, L. R. Hitney, and J. A. Ferguson, “ELF/VLF (Extremely Low
    Frequency/Very Low Frequency) long path pulse program for antennas of arbitrary
    elevation and orientation,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-891,
    Aug. 1983. [Online]. Available: http://www.dtic.mil/docs/citations/ADA133876.

[Pappert1986]: R. A. Pappert and J. A. Ferguson, “VLF/LF mode conversion model calculations
    for air to air transmissions in the earth-ionosphere waveguide,” Radio Sci., vol. 21,
    no. 4, pp. 551–558, Jul. 1986, doi: 10.1029/RS021i004p00551.
"""
function heightgains(z, ea₀, frequency, efconstants::ExcitationFactor; params=LMPParams())
    C, C² = ea₀.cosθ, ea₀.cos²θ
    k = frequency.k
    @unpack F₁, F₂, F₃, F₄, Rg = efconstants
    @unpack earthradius, earthcurvature = params

    if earthcurvature
        # Precompute
        α = 2/earthradius
        expz = exp(z/earthradius)  # assumes reflection coefficients are referenced to `d = 0`

        qz = pow23(k/α)*(C² + α*z)  # (k/α)^(2/3)*(C² + α*z)

        h₁z, h₂z, dh₁z, dh₂z = modifiedhankel(qz)

        # Precompute
        F₁h₁z = F₁*h₁z
        F₂h₂z = F₂*h₂z

        # Height gain for Ez, also called f∥(z).
        fz = expz*(F₁h₁z + F₂h₂z)

        # Height gain for Ey, also called f⟂(z)
        fy = (F₃*h₁z + F₄*h₂z)

        # Height gain for Ex, also called g(z)
        # f₂ = 1/(1im*k) df₁/dz
        fx = expz/(1im*k*earthradius)*(F₁h₁z + F₂h₂z + 2*pow23(k/α)*(F₁*dh₁z + F₂*dh₂z))

        # Height derivative of fy: d(fy)/dz = dqz/dz*(F₃*h₁′ + F₄*h₂′)
        # where qz = (k/α)^(2/3)*(C² + α*z) so dqz/dz = α*(k/α)^(2/3)
        dfy = α*pow23(k/α)*(F₃*dh₁z + F₄*dh₂z)
    else
        # Flat earth, [Pappert1983] pg. 12--13
        expiz = cis(k*C*z)
        fz = expiz + Rg[1,1]/expiz
        fy = expiz + Rg[2,2]/expiz
        fx = C*(expiz - Rg[1,1]/expiz)
        dfy = 1im*k*C*(expiz - Rg[2,2]/expiz)
    end

    return fz, fy, fx, dfy
end

"""
    radiationresistance(k, Cγ, zt)

Calculate radiation resistance correction for transmitting antenna elevated above the ground.
Based on [^Pappert1986] below, is derived from the time-averaged Poynting vector, which
uses total E and H fields calculated assuming a point dipole over perfectly reflecting ground.

If a point dipole at height z is radiating known power Pz, the power that should be input to
LMP is ``P/Pz = 2/f(kz,γ)``. Also note that E ∝ √P, hence the square root below.

# References

[Pappert1986]: R. A. Pappert, “Radiation resistance of thin antennas of arbitrary elevation
    and configuration over perfectly conducting ground.,” Naval Ocean Systems Center,
    San Diego, CA, Technical Report 1112, Jun. 1986. Accessed: Mar. 10, 2024. [Online].
    Available: https://apps.dtic.mil/sti/citations/ADA170945
"""
function radiationresistance(k, Cγ, zt)
    # TODO: Derive results for general Fresnel reflection coefficients.
    kz = 2*k*zt
    kz² = kz^2
    kz³ = kz²*kz

    sinkz, coskz = sincos(kz)
    Cγ² = Cγ^2
    Sγ² = 1 - Cγ²

    f = (1 + 3/kz³*(sinkz - kz*coskz))*Cγ² +
        (1 + 3/(2*kz³)*((1 - kz²)*sinkz - kz*coskz))*Sγ²

    corrfactor = sqrt(2/f)

    return corrfactor
end

@doc raw"""
    modeterms(modeequation, tx::Emitter, rx::AbstractSampler; params=LMPParams())

Compute `tx` and `rx` height-gain and excitation factor products and `ExcitationFactor`
constants returned as the tuple `(txterm, rxterm)`. Magnetic receiver terms in normalized
units (`Z₀H, V/m`).

The returned `txterm` is:
```math
λ_v \cos(γ) f_z(zₜ) + λ_b \sin(γ)\sin(ϕ) f_y(zₜ) + λ_e \sin(γ)\cos(ϕ) f_z(zₜ)
```
and `rxterm` is the height-gain function ``f(zᵣ)`` appropriate for `rx.fieldcomponent`:

| `fieldcomponent` |   ``f(zᵣ)``    |
|:----------------:|:--------------:|
|      ``Ez``       |  ``-S₀⋅f_z``          |
|      ``Ey``       |  ``EyHy⋅f_y``         |
|      ``Ex``       |     ``-f_x``          |
|      ``Hz``       |  ``-S₀⋅EyHy⋅f_y``     |
|      ``Hy``       |       ``f_z``        |
|      ``Hx``       |  ``EyHy⋅f_y′/(ik)``   |

# References

[Pappert1976]: R. A. Pappert and L. R. Shockey, “Simplified VLF/LF mode conversion program
    with allowance for elevated, arbitrarily oriented electric dipole antennas,” Naval
    Electronics Laboratory Center, San Diego, CA, Interim Report 771, Oct. 1976. [Online].
    Available: http://archive.org/details/DTIC_ADA033412.

[Morfitt1980]: D. G. Morfitt, “‘Simplified’ VLF/LF mode conversion computer programs:
    GRNDMC and ARBNMC,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-514, Jan. 1980.
    [Online]. Available: http://www.dtic.mil/docs/citations/ADA082695.
"""
function modeterms(modeequation, tx::Emitter, rx::AbstractSampler; params=LMPParams())
    @unpack ea, frequency, waveguide = modeequation
    @unpack ground = waveguide

    ea₀ = referencetoground(ea; params=params)
    S₀ = ea₀.sinθ

    frequency == tx.frequency ||
        throw(ArgumentError("`tx.frequency` and `modeequation.frequency` do not match"))

    zt = altitude(tx)
    zr = altitude(rx)

    # Transmit antenna orientation with respect to propagation direction
    # See [Morfitt1980] pg. 22
    Sγ, Cγ = sincos(inclination(tx))  # γ is measured from vertical
    Sϕ, Cϕ = sincos(azimuth(tx))  # ϕ is measured from `x`

    t1 = Cγ
    t2 = Sγ*Sϕ
    t3 = Sγ*Cϕ

    dFdθ, R, Rg = solvedmodalequation(modeequation; params=params)
    efconstants = excitationfactorconstants(ea₀, R, Rg, frequency, ground; params=params)

    λv, λb, λe = excitationfactor(ea, dFdθ, R, efconstants; params=params)

    # Transmitter term
    fzt, fyt, fxt, dfyt = heightgains(zt, ea₀, frequency, efconstants; params=params)
    txterm = λv*fzt*t1 + λb*fyt*t2 + λe*fxt*t3

    # Receiver term
    if zr == zt
        fzr, fyr, fxr, dfyr = fzt, fyt, fxt, dfyt
    else
        fzr, fyr, fxr, dfyr = heightgains(zr, ea₀, frequency, efconstants; params=params)
    end

    EyHy = efconstants.EyHy

    rxEz = -S₀*fzr
    rxEy = EyHy*fyr
    rxEx = -fxr

    rxHz = S₀*EyHy*fyr
    rxHy = fzr
    rxHx = EyHy*dfyr/(1im*frequency.k)
    #TODO confirm these are correct at non-zero altitudes with the index of refraction scaling for earth's curvature

    rxterm = SVector(rxEz, rxEy, rxEx, rxHz, rxHy, rxHx)

    return txterm, rxterm
end

# Specialized for the common case of `GroundSampler` and `Transmitter{VerticalDipole}`.
function modeterms(modeequation::ModeEquation, tx::Transmitter{VerticalDipole},
    rx::GroundSampler; params=LMPParams())

    @unpack ea, frequency, waveguide = modeequation
    @unpack ground = waveguide
    ea₀ = referencetoground(ea; params=params)
    S₀ = ea₀.sinθ

    frequency == tx.frequency ||
        throw(ArgumentError("`tx.frequency` and `modeequation.frequency` do not match"))

    dFdθ, R, Rg = solvedmodalequation(modeequation; params=params)
    efconstants = excitationfactorconstants(ea₀, R, Rg, frequency, ground; params=params)

    λv, _, _ = excitationfactor(ea, dFdθ, R, efconstants; params=params)

    # Transmitter term
    # TODO: specialized heightgains for z = 0
    fz, fy, fx, dfy = heightgains(0.0, ea₀, frequency, efconstants; params=params)
    txterm = λv*fz

    EyHy = efconstants.EyHy

    rxEz = -S₀*fz
    rxEy = EyHy*fy
    rxEx = -fx
    rxHz = S₀*EyHy*fy
    rxHy = fz
    rxHx = EyHy*dfy/(1im*frequency.k)
    rxterm = SVector(rxEz, rxEy, rxEx, rxHz, rxHy, rxHx)  # length == NUMFIELDCOMPONENTS

    return txterm, rxterm
end

#==
Electric field calculation
==#

"""
    fieldsum(modes, waveguide::HomogeneousWaveguide, tx::Emitter, rx::AbstractSampler;
           params=LMPParams())

Compute the complex electric field by summing `modes` in `waveguide` for emitter `tx` at
sampler `rx`.

This function always returns all three electric field components, regardless of the value
of `rx.fieldcomponent`.

# References

[Morfitt1980]: D. G. Morfitt, “‘Simplified’ VLF/LF mode conversion computer programs:
    GRNDMC and ARBNMC,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-514, Jan. 1980.
    [Online]. Available: http://www.dtic.mil/docs/citations/ADA082695.

[Pappert1983]: R. A. Pappert, L. R. Hitney, and J. A. Ferguson, “ELF/VLF (Extremely Low
    Frequency/Very Low Frequency) long path pulse program for antennas of arbitrary
    elevation and orientation,” Naval Ocean Systems Center, San Diego, CA, NOSC/TR-891,
    Aug. 1983. [Online]. Available: http://www.dtic.mil/docs/citations/ADA133876.
"""
function fieldsum(modes, waveguide::HomogeneousWaveguide, tx::Emitter, rx::AbstractSampler;
    params=LMPParams())

    # There's no compute time advantage switching to a specialized version of fieldsum for
    # scalars.

    X = distance(rx, tx)

    # Restricting dimensions and computation to `numcomponents(fc)` provides negligible
    # computation gains relative to the modefinder. Likewise using a vector of MVectors.
    E = zeros(ComplexF64, NUMFIELDCOMPONENTS, length(X))

    txpower = power(tx)
    frequency = tx.frequency
    k = frequency.k
    zt = altitude(tx)
    Cγ = cos(inclination(tx))

    for ea in modes
        modeequation = PhysicalModeEquation(ea, frequency, waveguide)
        txterm, rxterm = modeterms(modeequation, tx, rx; params=params)

        S₀ = referencetoground(ea.sinθ; params=params)
        expterm = -k*(S₀ - 1)
        txrxterm = txterm.*rxterm

        for i in axes(E,2)
            @. E[:,i] += txrxterm*cis(expterm*X[i])
        end
    end

    Q = 0.6822408*sqrt(frequency.f*txpower)  # factor from lw_sum_modes.for
    # Q = Z₀/(4π)*sqrt(2π*txpower/10k)*k/2  # Ferguson and Morfitt 1981 eq (21), V/m, NOT uV/m!
    # Q *= 100 # for V/m to uV/m

    if params.radiationresistancecorrection && zt > 0
        corrfactor = radiationresistance(k, Cγ, zt)
        Q *= corrfactor
    end

    for i in axes(E,2)
        @. E[:,i] *= Q/sqrt(abs(sin(X[i]/params.earthradius)))
    end

    return E
end

function fieldsum(modes, waveguide::HomogeneousWaveguide, tx::Emitter,
    rx::AbstractSampler{<:Real}; params=LMPParams())

    frequency = tx.frequency
    k = frequency.k
    zt = altitude(tx)
    Cγ = cos(inclination(tx))
    txpower = power(tx)

    x = distance(rx, tx)

    Q = 0.6822408*sqrt(frequency.f*txpower)

    if params.radiationresistancecorrection && zt > 0
        corrfactor = radiationresistance(k, Cγ, zt)
        Q *= corrfactor
    end

    E = zeros(ComplexF64, NUMFIELDCOMPONENTS)
    for ea in modes
        modeequation = PhysicalModeEquation(ea, frequency, waveguide)
        txterm, rxterm = modeterms(modeequation, tx, rx, params=params)

        S₀ = referencetoground(ea.sinθ; params=params)
        expterm = -k*(S₀ - 1)
        txrxterm = txterm.*rxterm

        @. E += txrxterm*cis(expterm*x)
    end

    @. E *= Q/sqrt(abs(sin(x/params.earthradius)))

    return E
end

"""
    fieldsum(waveguide::SegmentedWaveguide, wavefields_vec, adjwavefields_vec, tx::Emitter,
           rx::AbstractSampler; params=LMPParams())
"""
function fieldsum(waveguide::SegmentedWaveguide, wavefields_vec, adjwavefields_vec,
    tx::Emitter, rx::AbstractSampler; params=LMPParams())
    @unpack earthradius = params

    # Checks
    first(waveguide).distance == 0 ||
        throw(ArgumentError("The first `waveguide` segment should have `distance` 0.0."))
    length(waveguide) == length(wavefields_vec) == length(adjwavefields_vec) ||
        throw(ArgumentError("`wavefields_vec` and `adjwavefields_vec` must have the same"*
                            "length as `waveguide`."))

    X = distance(rx, tx)
    maxX = maximum(X)
    Xlength = length(X)
    E = Matrix{ComplexF64}(undef, NUMFIELDCOMPONENTS, Xlength)

    frequency = tx.frequency
    k = frequency.k
    zt = altitude(tx)
    Cγ = cos(inclination(tx))

    Q = 0.6822408*sqrt(frequency.f*tx.power)

    if params.radiationresistancecorrection && zt > 0
        corrfactor = radiationresistance(k, Cγ, zt)
        Q *= corrfactor
    end

    # Initialize
    J = length(waveguide)
    M = 0  # number of eigenangles in previous segment. Current segment is N
    xmtrfields = Vector{ComplexF64}(undef, 0)  # fields generated by transmitter
    previous_xmtrfields = similar(xmtrfields)  # fields saved from previous segment
    rcvrfields = Vector{SVector{NUMFIELDCOMPONENTS, ComplexF64}}(undef, 0)  # fields at receiver location

    i = 1  # index of X
    for j = 1:J  # index of waveguide
        wvg = waveguide[j]
        wavefields = wavefields_vec[j]
        eas = eigenangles(wavefields)
        N = nummodes(wavefields)

        # Identify distance at beginning of segment
        segment_start = wvg.distance
        maxX < segment_start && break  # no farther X; break

        # Identify distance at end of segment
        if j < J
            segment_end = waveguide[j+1].distance
        else
            # last segment
            segment_end = typemax(typeof(segment_start))
        end

        # xmtrfields is for `Hy`
        resize!(xmtrfields, N)
        resize!(rcvrfields, N)
        if j > 1
            adjwavefields = adjwavefields_vec[j]
            prevwavefields = wavefields_vec[j-1]
            conversioncoeffs = modeconversion(prevwavefields, wavefields, adjwavefields;
                                              params=params)
        end

        # Calculate the mode terms (height gains and excitation factors) up to the current
        # segment
        for n = 1:N
            modeequation = PhysicalModeEquation(eas[n], frequency, wvg)
            txterm, rxterm = modeterms(modeequation, tx, rx; params=params)
            if j == 1
                # Transmitter exists only in the transmitter slab (obviously)
                xmtrfields[n] = txterm
            else
                # Otherwise, mode conversion of transmitted fields
                xmtrfields_sum = zero(eltype(xmtrfields))
                for m = 1:M
                    xmtrfields_sum += previous_xmtrfields[m]*conversioncoeffs[m,n]
                end
                xmtrfields[n] = xmtrfields_sum
            end

            rcvrfields[n] = xmtrfields[n].*rxterm
        end

        # Calculate E at each distance in the current waveguide segment
        while X[i] < segment_end
            x = X[i] - segment_start
            factor = Q/sqrt(abs(sin(X[i]/earthradius)))

            totalfield = zeros(MVector{NUMFIELDCOMPONENTS, eltype(E)})
            for n = 1:N
                S₀ = referencetoground(eas[n].sinθ; params=params)
                totalfield .+= rcvrfields[n]*cis(-k*x*(S₀ - 1))*factor
            end

            E[:,i] .= totalfield
            i += 1
            i > Xlength && break
        end

        # If we've reached the end of the current segment and there are more segments,
        # prepare for next segment
        if j < J
            # End of current slab
            x = segment_end - segment_start

            resize!(previous_xmtrfields, N)
            for n = 1:N
                S₀ = referencetoground(eas[n].sinθ; params=params)

                # Excitation factors at end of slab
                xmtrfields[n] *= cis(-k*x*(S₀ - 1))
                previous_xmtrfields[n] = xmtrfields[n]
            end
            M = N  # set previous number of modes
        end
    end

    return E
end
