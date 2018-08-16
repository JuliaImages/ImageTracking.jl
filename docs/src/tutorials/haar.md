# Haar-like features

Haar-like features are digital image features used in object recognition. They owe their name to their intuitive similarity with Haar
wavelets and were used in the first real-time face detector developed by Viola and Jones.

A simple rectangular Haar-like feature can be defined as the difference of the sum of pixels of areas inside the rectangle, which can be at any position
and scale within the original image. This modified feature set is called 2-rectangle feature. Viola and Jones also defined 3-rectangle features and
4-rectangle features. The values indicate certain characteristics of a particular area of the image. Each feature type can indicate the existence (or
absence) of certain characteristics in the image, such as edges or changes in texture. For example, a 2-rectangle feature can indicate where the border
lies between a dark region and a light region.

The ImageTracking package houses two function related to haar-like features:

`haar_features`       - Returns an array containing the Haar-like features for the given `Integral Image` in the region specified by the points `top_left`
and `bottom_right`.
`haar_coordinates`    - Returns an array containing the `coordinates` of all the possible Haar-like features of the specified type in any region of given
`height` and `width`.
