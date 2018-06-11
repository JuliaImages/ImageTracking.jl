"""
    Farneback(Args...)

A method for dense optical flow estimation developed by Gunnar Farneback. It
computes the optical flow for all the points in the frame using the polynomial
representation of the images. The idea of polynomial expansion is to approximate
the neighbourhood of a point in a 2D function with a polynomial. Displacement
fields are estimated from the polynomial coefficients depending on how the
polynomial transforms under translation.

The different arguments are:

 -  flow_est          =  Array of SVector{2} containing estimate flow values for all points in the frame
 -  iterations        =  Number of iterations the displacement estimation algorithm is run at each
                         point
 -  window_size       =  Size of the search window at each pyramid level; the total size of the
                         window used is 2*window_size + 1
 -  σw                =  Standard deviation of the Gaussian weighting filter
 -  neighbourhood     =  size of the pixel neighbourhood used to find polynomial expansion for each pixel;
                         larger values mean that the image will be approximated with smoother surfaces,
                         yielding more robust algorithm and more blurred motion field
 -  σp                =  standard deviation of the Gaussian that is used to smooth derivatives used as a
                         basis for the polynomial expansion (Applicability)
 -  est_flag          =  true -> Use flow_est as initial estimate
                         false -> Assume zero initial flow values
 -  gauss_flag        =  false -> use box filter
                         true -> use gaussian filter instead of box filter of the same size for optical flow
                         estimation; usually, this option gives more accurate flow than with a box filter,
                         at the cost of lower speed (Default Value)

## References

Farnebäck G. (2003) Two-Frame Motion Estimation Based on Polynomial Expansion. In: Bigun J.,
Gustavsson T. (eds) Image Analysis. SCIA 2003. Lecture Notes in Computer Science, vol 2749. Springer, Berlin,
Heidelberg

Farnebäck, G.: Polynomial Expansion for Orientation and Motion Estimation. PhD thesis, Linköping University,
Sweden, SE-581 83 Linköping, Sweden (2002) Dissertation No 790, ISBN 91-7373-475-6.

"""

struct Farneback{F <: Float64, I <: Int64, V <: Bool} <: OpticalFlowAlgo
    flow_est::Array{SVector{2, F}, 2}
    iterations::I
    window_size::I
    σw::F
    neighbourhood::I
    σp::F
    est_flag::V
    gauss_flag::V
end

Farneback(flow_est::Array{SVector{2, F}, 2}, iterations::I, window_size::I, σw::F, neighbourhood::I, σp::F, est_flag::V) where {F <: Float64, I <: Int64, V <: Bool} = LK{F, I, V}(flow_est, iterations, window_size, σw, neighbourhood, σp, est_flag, true)

function optflow(first_img::AbstractArray{T, 2}, second_img::AbstractArray{T,2}, algo::Farneback{}) where T <: Gray
    if algo.est_flag
        @assert size(first_img) == size(algo.flow_est)
    end
    A1, B1, _ = polynomial_expansion(first_img, algo.neighbourhood, algo.σp)
    A2, B2, _ = polynomial_expansion(second_img, algo.neighbourhood, algo.σp)

    AtA = similar(A1)
    AtB = similar(B1)

    disp = Array{MVector{2, Float64}, 2}(size(first_img))
    for i = 1:size(first_img)[1]
        for j = 1:size(first_img)[2]
            disp[i,j] = MVector{2, Float64}(0.0,0.0)
        end
    end

    for k = 1:algo.iterations
        for i = 1:size(first_img)[1]
            for j = 1:size(first_img)[2]
                d = disp[i,j][:]

                mod_i = round(Int, i + d[2])
                mod_j = round(Int, j + d[1])

                mod_i = clamp(mod_i, 1, size(first_img)[1])
                mod_j = clamp(mod_j, 1, size(first_img)[2])

                sub_A1 = A1[i,j,:,:]
                sub_B1 = B1[i,j,:]
                sub_A2 = A2[mod_i,mod_j,:,:]
                sub_B2 = B2[mod_i,mod_j,:]

                A = (sub_A1 .+ sub_A2)./2
                dB = A*d .- ((sub_B2 .- sub_B1)./2)

                AtA[i,j,:,:] = A'*A
                AtB[i,j,:] = A'*dB
            end
        end

        len = floor(Int, algo.window_size/2)
        w = ImageFiltering.KernelFactors.gaussian((algo.σw),(algo.window_size))[-len:len]

        for i = 1:2
            for j = 1:2
                if algo.gauss_flag
                    AtA[:,:,i,j] = conv2(w, w, AtA[:,:,i,j])[1+len:size(first_img)[1]+len,1+len:size(first_img)[2]+len]
                else
                    AtA[:,:,i,j] = conv2(ones(algo.window_size)./algo.window_size, ones(algo.window_size)./algo.window_size, AtA[:,:,i,j])[1+len:size(first_img)[1]+len,1+len:size(first_img)[2]+len]
                end
            end
            if algo.gauss_flag
                AtB[:,:,i] = conv2(w, w, AtB[:,:,i])[1+len:size(first_img)[1]+len,1+len:size(first_img)[2]+len]
            else
                AtB[:,:,i,j] = conv2(ones(algo.window_size)./algo.window_size, ones(algo.window_size)./algo.window_size, AtB[:,:,i,j])[1+len:size(first_img)[1]+len,1+len:size(first_img)[2]+len]
            end
        end

        for i = 1:size(first_img)[1]
            for j = 1:size(first_img)[2]
                disp[i,j][:] = MVector{2,Float64}(pinv(AtA[i,j,:,:])*AtB[i,j,:])
            end
        end
    end

    return disp
end

"""
```
A, B, C = polynomial_expansion(img, neighbourhood, σ)
```

Returns the polynomial coefficients of the approximation of the neighbourhood of a
point in a 2D function with a polynomial.
The expansion is: f(x) = x'Ax + B'x + C

Parameters:

-  img               =  Grayscale image who polynomial expansion is to be found
-  neighbourhood     =  Size of the pixel neighbourhood used to find polynomial expansion for each pixel;
                        larger values mean that the image will be approximated with smoother surfaces,
                        yielding more robust algorithm and more blurred motion field
-  σ                 =  Standard deviation of the Gaussian that is used to smooth derivatives used as a
                        basis for the polynomial expansion (Applicability)

## References

Farnebäck, G.: Polynomial Expansion for Orientation and Motion Estimation. PhD thesis, Linköping University,
Sweden, SE-581 83 Linköping, Sweden (2002) Dissertation No 790, ISBN 91-7373-475-6.

"""

function polynomial_expansion(img::AbstractArray{T, 2}, neighbourhood::Int, σ::Real) where T <: Gray
    len = floor(Int, neighbourhood/2)
    a = ImageFiltering.Kernel.gaussian((σ,σ), (neighbourhood,neighbourhood))[-len:len, -len:len]

    c = ones(size(img))
    padded_c = reshape(padarray(c, Fill(0, (len, len)))[:], size(img) .+ 2*len)

    padded_img = reshape(padarray(img, Fill(0, (len, len)))[:], size(img) .+ 2*len)

    B = zeros(neighbourhood*neighbourhood, 6)
    for x = -len:len
        for y = -len:len
            B[neighbourhood*(x + len) + (y + len) + 1, :] = [1, x, y, x*x, y*y, x*y]
        end
    end

    Wa = diagm(a[:])
    BtWa = B'*Wa

    poly_A = zeros(size(img)[1], size(img)[2], 2, 2)
    poly_B = zeros(size(img)[1], size(img)[2], 2)
    poly_C = zeros(size(img))

    for i = (len + 1):(size(img)[1] - len)
        for j = (len + 1):(size(img)[2] - len)
            sub_c = padded_c[(i - len):(i + len), (j - len):(j + len)]
            sub_img = padded_img[(i - len):(i + len), (j - len):(j + len)]

            Wc = diagm(sub_c[:])
            BtWaWc = BtWa*Wc
            G = BtWaWc*B
            G_inv = pinv(G)
            r = G_inv*BtWaWc*sub_img[:]

            poly_A[i-len, j-len, :, :] = [r[4] r[6]/2
                                          r[6]/2 r[5]]

            poly_B[i-len, j-len, :] = [r[2], r[3]]

            poly_C[i-len, j-len] = r[1]
        end
    end

    return (poly_A, poly_B, poly_C)
end
