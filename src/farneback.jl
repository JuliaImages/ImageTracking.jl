

abstract type AbstractImplementation end
struct MatrixImplementation <: AbstractImplementation end
struct ConvolutionImplementation <: AbstractImplementation end

function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2},  algorithm::Farneback{}) where T <: Gray
    displacement = Array{SVector{2, Float64}, 2}(undef,size(first_img))
    for i in eachindex(displacement)
            displacement[i] = SVector{2, Float64}(0.0, 0.0)
    end
    optflow!(first_img, second_img, displacement,  algorithm)
end

function optflow!(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, displacement::Array{SVector{2, Float64}, 2}, algorithm::Farneback{}) where T <: Gray

    # Replace NaN with zero in both images.
    map!(x -> isnan(x) ? zero(x) : x, first_img, first_img)
    map!(x -> isnan(x) ? zero(x) : x, second_img, second_img)

    start_rows = first(first(axes(first_img)))
    start_cols = first(last(axes(first_img)))
    end_rows = last(first(axes(first_img)))
    end_cols = last(last(axes(first_img)))

    A1, B1, _ = polynomial_expansion(ConvolutionImplementation(), first_img, algorithm.expansion_window, algorithm.σ_expansion_window)
    A2, B2, _ = polynomial_expansion(ConvolutionImplementation(), second_img, algorithm.expansion_window, algorithm.σ_expansion_window)

    AtA = zero(similar(A1))
    AtB = zero(similar(B1))

    AtA_smoothed = zero(similar(A1))
    AtB_smoothed = zero(similar(B1))

    neighbourhood = 2*algorithm.estimation_window + 1
    σ, w =  (algorithm.σ_estimation_window,algorithm.σ_estimation_window), (neighbourhood,neighbourhood)
    kern = Kernel.gaussian(σ ,w)

    Aₗ = zeros(Float64,2,2)
    AtAₗ = zeros(Float64,2,2)
    AtBₗ = zeros(Float64,2,1)
    Adₗ = zeros(Float64,2,1)
    bₗ = zeros(Float64,2,1)
    dbₗ = zeros(Float64,2,1)
    dₗ = zeros(Float64,2,1)
    len =  algorithm.expansion_window
    for k = 1:algorithm.iterations
        # We don't utilise the raw polynomial expansion coefficients at the
        # border of the image; hence, we add and subtract `len`.
        for j = start_cols + len:end_cols - len
            for i = start_rows + len:end_rows - len
                d = displacement[i,j]

                mod_i = round(Int, i + d[2])
                mod_j = round(Int, j + d[1])
                mod_i = clamp(mod_i, 1, first(size(first_img)))
                mod_j = clamp(mod_j, 1, last(size(first_img)))

                # (A1[i,j,:,:] + A2[i,j,:,:]) / 2
                for p = 1:2
                    for q = 1:2
                        setindex!(Aₗ, A1[i,j,p,q] / 2 + A2[mod_i,mod_j,p,q] / 2, p, q)
                    end
                end
                # A*d
                setindex!(dₗ, d[1], 1)
                setindex!(dₗ, d[2], 2)
                mul!(Adₗ, Aₗ, dₗ)
                # (B2[i,j,:] - B1[i,j,:]) / 2
                for p = 1:2
                    setindex!(bₗ, (B2[mod_i,mod_j,p] - B1[i,j,p]) / 2 , p)
                end
                # dB = A*d - ((B2 - B1)/2)
                broadcast!(-, dbₗ, Adₗ, bₗ)
                # A'*A
                mul!(AtAₗ,Aₗ',Aₗ)
                # A'*dB
                mul!(AtBₗ,Aₗ',dbₗ)
                # Store result
                for p = 1:2
                    for q = 1:2
                        setindex!(AtA, AtAₗ[p,q], i, j, p, q)
                    end
                    setindex!(AtB, AtBₗ[p], i, j, p)
                end
            end
        end

        for p = 1:2
            for q = 1:2
                AtAview = @view AtA[:,:,p,q]
                AtAview_smoothed = @view AtA_smoothed[:,:,p,q]
                imfilter!(AtAview_smoothed, AtAview, kern, Pad(:replicate))
            end
            AtBview = @view AtB[:,:,p]
            AtBview_smoothed = @view AtB_smoothed[:,:,p]
            imfilter!(AtBview_smoothed, AtBview, kern, Pad(:replicate))
        end

        for j = start_cols:end_cols
            for i = start_rows:end_rows
                P = @view AtA_smoothed[i,j,:,:]
                Q = @view  AtB_smoothed[i,j,:]
                displacement[i,j] = SVector{2, Float64}(pinv2x2(P) * Q)
            end
        end
    end
    # Convert (x,y) coordinate convention to (row, col) convention.
    map!(x -> SVector(last(x), first(x)), displacement, displacement)
    return displacement
end


"""
```
A, B, C = polynomial_expansion(implementation, img, neighbourhood, σ)
```

Returns the polynomial coefficients of the approximation of the neighbourhood of a
point in a 2D function with a polynomial.
The expansion is: f(x) = x'Ax + B'x + C

# Options

Various options for the fields of this type are described in more detail
below.

## Choices for `implementation`

Selects implementation (`::MatrixImplementation` vs `::ConvolutionImplementation`).
The matrix implementation serves as a useful reference against which other
faster implementations can be validated.

## Choices for `img`

Grayscale (Float) image whose polynomial expansion is to be found.

## Choices for `window_size`

Determines the size of the pixel neighbourhood used to find polynomial expansion
for each pixel; larger values mean that the image will be approximated with
smoother surfaces, yielding more robust algorithm and more blurred motion field.
The total size equals `2*window_size + 1`.

## Choices for `σ`

Standard deviation of the Gaussian that is used to smooth the image for the purpose
of approximating it with a polynomial expansion.


## References

Farnebäck, G.: Polynomial Expansion for Orientation and Motion Estimation. PhD thesis, Linköping University,
Sweden, SE-581 83 Linköping, Sweden (2002) Dissertation No 790, ISBN 91-7373-475-6.

"""
function polynomial_expansion(implementation::MatrixImplementation, img::AbstractArray{T, 2}, window_size::Int, σ::Real) where T <: Gray
    len = window_size
    neighbourhood = 2*window_size + 1
    a = ImageFiltering.Kernel.gaussian((σ,σ), (neighbourhood,neighbourhood))[-len:len, -len:len]

    padded_img = parent(padarray(img, Fill(0, (len, len))))

    B0 = zeros(neighbourhood*neighbourhood, 6)
    for x = -len:len
        for y = -len:len
            B0[neighbourhood*(x + len) + (y + len) + 1, :] = [1, x, y, x*x, y*y, x*y]
        end
    end

    B = SMatrix{neighbourhood*neighbourhood, 6}(B0)
    Wa = SDiagonal{neighbourhood^2,Float64}( @view a[:])
    BtWa = B'*Wa

    # If we assume that the confidence is constant we can move several
    # matrix multiplications out of the main loop.
    Wc = SDiagonal(ones(1,neighbourhood^2)...)
    BtWaWc = BtWa*Wc
    G = BtWaWc*B
    G_inv = SMatrix{6,6,Float64}(pinv(convert(Array,G)))
    G_invBtWaWc = G_inv*BtWaWc
    poly_A = zeros(size(img)[1], size(img)[2], 2, 2)
    poly_B = zeros(size(img)[1], size(img)[2], 2)
    poly_C = zeros(size(img))
    f = @MVector zeros(neighbourhood^2)
    for i = (len + 1):(size(padded_img)[1] - len)
        for j = (len + 1):(size(padded_img)[2] - len)
            sub_img = @view padded_img[(i - len):(i + len), (j - len):(j + len)]
            f .= vec(sub_img)
            r = G_invBtWaWc*f

            poly_A[i-len, j-len, 1, 1] =  r[4]
            poly_A[i-len, j-len, 2, 1] =  r[6]/2
            poly_A[i-len, j-len, 1, 2] =  r[6]/2
            poly_A[i-len, j-len, 2, 2] =  r[5]

            poly_B[i-len, j-len, 1] = r[2]
            poly_B[i-len, j-len, 2] = r[3]

            poly_C[i-len, j-len] = r[1]
        end
    end
    return (poly_A, poly_B, poly_C)
end

function polynomial_expansion(imp::ConvolutionImplementation, img::AbstractArray{T, 2}, window_size::Int, σ::Real) where T <: Gray
    #len = floor(Int, neighbourhood/2)
    len = window_size
    neighbourhood = 2*window_size + 1

    # Define basis functions.
    P₂ = construct_basis(len, neighbourhood)

    # Construct Gaussian filter.
    g = collect(KernelFactors.gaussian(σ,neighbourhood))

    # Construct inverse metric using Gaussian "applicability".
    G⁻¹ = construct_inverse_metric(P₂, g)

    unity, x, x², y, xy, y² = calculate_correlations(img, len, g)

    # In-place version of the following code (thus reducing allocations):
    # unity_tmp = G⁻¹[1,1] * unity + G⁻¹[1,4] * x² + G⁻¹[1,5] * y²
    # x = G⁻¹[2,2] * x
    # y = G⁻¹[3,3] * y
    # x² = G⁻¹[4,4] * x² + G⁻¹[4,1] * unity
    # y² = G⁻¹[5,5] * y² + G⁻¹[5,1] * unity
    # xy = G⁻¹[6,6] * xy
    # unity = unity_tmp
    unity_tmp = similar(unity)
    broadcast!(+,unity_tmp, lmul!(G⁻¹[1,1], unity), lmul!(G⁻¹[1,4], x²), lmul!(G⁻¹[1,5] , y²))
    x =  lmul!(G⁻¹[2,2], x)
    y =  lmul!(G⁻¹[3,3], y)
    x² = broadcast!(+,x², lmul!(G⁻¹[4,4]/G⁻¹[1,4], x²), lmul!(G⁻¹[4,1]/G⁻¹[1,1], unity))
    y² = broadcast!(+,y², lmul!(G⁻¹[5,5]/G⁻¹[1,5], y² ), lmul!(G⁻¹[5,1]/G⁻¹[4,1], unity))
    xy = lmul!(G⁻¹[6,6], xy)
    unity = unity_tmp

    # # The result of each multiplication allocated memory which is undesirable.
    # # I've left this version because I am trying to track down an occasional
    # # NaN bug and I want to be sure it is not due to the `in-place` version of
    # # this code.
    # unity_tmp = G⁻¹[1,1] * unity + G⁻¹[1,4] * x² + G⁻¹[1,5] * y²
    # x = G⁻¹[2,2] * x
    # y = G⁻¹[3,3] * y
    # x² = G⁻¹[4,4] * x² + G⁻¹[4,1] * unity
    # y² = G⁻¹[5,5] * y² + G⁻¹[5,1] * unity
    # xy = G⁻¹[6,6] * xy
    # unity = unity_tmp


    poly_A = zeros(Float64,first(size(img)), last(size(img)), 2, 2)
    poly_B = zeros(Float64,first(size(img)), last(size(img)), 2)
    poly_C = unity
    for j =  1:last(size(img))
        for i = 1:first(size(img))
            @inbounds setindex!(poly_A, x²[i,j], i, j, 1, 1)
            @inbounds setindex!(poly_A, xy[i,j]/2, i, j, 1, 2)
            @inbounds setindex!(poly_A, poly_A[i,j,1,2], i, j, 2, 1)
            @inbounds setindex!(poly_A, y²[i,j], i, j, 2, 2)
            @inbounds setindex!(poly_B, x[i,j], i, j, 1)
            @inbounds setindex!(poly_B, y[i,j], i, j, 2)
        end
    end
    return (poly_A, poly_B, poly_C)
end

function construct_basis(len::Int, neighbourhood::Int)
    P₂ = zeros(neighbourhood*neighbourhood, 6)
    for x = -len:len
        for y = -len:len
            P₂[neighbourhood*(x + len) + (y + len) + 1, :] = [1, x, y, x*x, y*y, x*y]
        end
    end
    return P₂
end

function construct_inverse_metric(P₂::AbstractArray, g::AbstractArray)
    applicability = g*g'
    Wa = Matrix(Diagonal(@view applicability[:]))
    P₂_Wa_P₂ = P₂'*Wa*P₂
    G⁻¹ = pinv(P₂_Wa_P₂)
end

function calculate_correlations(img, len, g)
    # Define basis functions.
    l = collect(-len:len)
    lg = l.*g
    l²g = l.*lg


    # Correlations along the first dimensions.
    unity_first_dimension = imfilter(img, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(0))
    y_first_dimension = imfilter(img, ( ImageFiltering.ReshapedOneD{2,0}(centered(lg)),),  Fill(0))
    y²_first_dimension = imfilter(img, ( ImageFiltering.ReshapedOneD{2,0}(centered(l²g)),), Fill(0))

    # Correlations along the second dimensions.
    # Figure 4.1 (Follow correlation structure) 1, y, y², x , xy, x²
    unity = imfilter(unity_first_dimension, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(0))
    x =  imfilter(unity_first_dimension, ( ImageFiltering.ReshapedOneD{2,1}(centered(lg)),), Fill(0))
    x² =  imfilter(unity_first_dimension, ( ImageFiltering.ReshapedOneD{2,1}(centered(l²g)),), Fill(0))

    y =  imfilter(y_first_dimension, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(0))
    xy = imfilter(y_first_dimension, ( ImageFiltering.ReshapedOneD{2,1}(centered(lg)),), Fill(0))

    y² = imfilter(y²_first_dimension, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(0))

    return unity, x, x², y, xy, y²
end
