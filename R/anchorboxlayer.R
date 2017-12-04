#' @export
AnchorBoxesLayer <- R6::R6Class("KerasLayer",

                           inherit = KerasLayer,

                           public = list(
                             img_height = NULL,
                             img_width = NULL,
                             this_scale = NULL,
                             next_scale = NULL,
                             aspect_ratios = c(0.5, 1.0, 2.0),
                             two_boxes_for_ar1 = TRUE,
                             limit_boxes = TRUE,
                             variances = c(1.0, 1.0, 1.0, 1.0),
                             coords = 'centroids',
                             normalize_coords = FALSE,
                             n_boxes = NULL,
                             input_shape = NULL,

                             initialize = function(img_height,
                                                   img_width,
                                                   this_scale,
                                                   next_scale,
                                                   aspect_ratios = c(0.5, 1.0, 2.0),
                                                   two_boxes_for_ar1 = TRUE,
                                                   limit_boxes = TRUE,
                                                   variances = c(1.0, 1.0, 1.0, 1.0),
                                                   coords = 'centroids',
                                                   normalize_coords = FALSE,
                                                   ...) {
                               if (K$backend() != 'tensorflow') {
                                 stop("error")
                                 # stop("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))
                               }
                               if ((this_scale < 0) || (next_scale < 0) || (this_scale > 1)) {
                                 stop("error")
                                 # stop("`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(this_scale, next_scale))
                               }
                               if (length(variances) != 4) {
                                 stop("error")
                                 # raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
                               }
                               # variances = np.array(variances)
                               if (any(variances <= 0)) {
                                 stop("error")
                                 # raise ValueError("All variances must be >0, but the variances given are {}".format(variances))
                               }

                               self$img_height = img_height
                               self$img_width = img_width
                               self$this_scale = this_scale
                               self$next_scale = next_scale
                               self$aspect_ratios = aspect_ratios
                               self$two_boxes_for_ar1 = two_boxes_for_ar1
                               self$limit_boxes = limit_boxes
                               self$variances = variances
                               self$coords = coords
                               self$normalize_coords = normalize_coords

                               # Compute the number of boxes per cell
                               if (1 %in% aspect_ratios && two_boxes_for_ar1) {
                                 self$n_boxes = length(aspect_ratios) + 1L
                               } else {
                                 self$n_boxes = length(aspect_ratios)
                               }
                               #super$initialize(...)

                             },

                             # build = function(input_shape) {
                             #   self$input_shape = input_shape
                             #   super$build(input_shape)
                             # },

                             call = function(x, mask = NULL) {
                               np <- reticulate::import("numpy")
                               # Compute box width and height for each aspect ratio
                               # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
                               self$aspect_ratios = sort(self$aspect_ratios)
                               size = min(self$img_height, self$img_width)
                               # Compute the box widths and and heights for all aspect ratios
                               wh_list = data.frame(w = double(0), h = double(0))
                               for (ar in self$aspect_ratios) {
                                 if ((ar == 1) && self$two_boxes_for_ar1) {
                                   # Compute the regular default box for aspect ratio 1 and...
                                   w = self$this_scale * size * sqrt(ar)
                                   h = self$this_scale * size / sqrt(ar)
                                   wh_list <- rbind.data.frame(wh_list, data.frame(w,h))
                                   # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                                   w = sqrt(self$this_scale * self$next_scale) * size * sqrt(ar)
                                   h = sqrt(self$this_scale * self$next_scale) * size / sqrt(ar)
                                   wh_list <- rbind.data.frame(wh_list, data.frame(w,h))
                                 } else {
                                   w = self$this_scale * size * sqrt(ar)
                                   h = self$this_scale * size / sqrt(ar)
                                   wh_list <- rbind.data.frame(wh_list, data.frame(w,h))
                                 }
                               }
                               # wh_list = np.array(wh_list)

                               # We need the shape of the input tensor
                               if (K$image_dim_ordering() == 'tf') {
                                 batch_size = x$shape$dims[[1]]$value
                                 feature_map_height = x$shape$dims[[2]]$value
                                 feature_map_width = x$shape$dims[[3]]$value
                                 feature_map_channels = x$shape$dims[[4]]$value
                               } else {
                                 batch_size = x$shape$dims[[1]]$value
                                 feature_map_channels = x$shape$dims[[2]]$value
                                 feature_map_height = x$shape$dims[[3]]$value
                                 feature_map_width = x$shape$dims[[4]]$value
                               }

                               # Compute the grid of box center points. They are identical for all aspect ratios
                               cell_height = self$img_height / feature_map_height
                               cell_width = self$img_width / feature_map_width
                               cx = seq(cell_width/2, self$img_width-cell_width/2, length.out = feature_map_width)
                               cy = seq(cell_height/2, self$img_height-cell_height/2, length.out = feature_map_height)
                               grid = np$meshgrid(cx, cy)
                               # cx_grid = matrix(grid[, 1], feature_map_height, feature_map_width)
                               # cy_grid = matrix(grid[, 2], feature_map_height, feature_map_width)
                               cx_grid = grid[[1]]
                               cy_grid = grid[[2]]
                               cx_grid = np$expand_dims(cx_grid, -1L)
                               cy_grid = np$expand_dims(cy_grid, -1L)
                               # cx_grid = array(t(cx_grid), dim = c(feature_map_height, feature_map_width, 1)) # This is necessary for np.tile() to do what we want further down
                               # cy_grid = array(t(cy_grid), dim = c(feature_map_height, feature_map_width, 1)) # This is necessary for np.tile() to do what we want further down

                               # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
                               # where the last dimension will contain `(cx, cy, w, h)`
                               # boxes_tensor = array(0, dim = c(feature_map_height, feature_map_width, self$n_boxes, 4))
                               boxes_tensor = np$zeros(reticulate::tuple(feature_map_height, feature_map_width, self$n_boxes, 4L))

                               # boxes_tensor[, , , 1] = array(cx_grid, dim = c(feature_map_height, feature_map_width, self$n_boxes)) #np.tile(cx_grid, (1, 1, self$n_boxes)) # Set cx
                               boxes_tensor[, , , 1] = np$tile(cx_grid, reticulate::tuple(1L, 1L, self$n_boxes)) # Set cx
                               # boxes_tensor[, , , 2] = array(cy_grid, dim = c(feature_map_height, feature_map_width, self$n_boxes)) #np.tile(cy_grid, (1, 1, self$n_boxes)) # Set cy
                               boxes_tensor[, , , 2] = np$tile(cy_grid, reticulate::tuple(1L, 1L, self$n_boxes)) # Set cx
                               boxes_tensor[, , , 3] = array(rep(wh_list[, 1], each = feature_map_height * feature_map_width), dim = c(feature_map_height, feature_map_width, self$n_boxes)) # Set w
                               boxes_tensor[, , , 4] = array(rep(wh_list[, 2], each = feature_map_height * feature_map_width), dim = c(feature_map_height, feature_map_width, self$n_boxes)) # Set h

                               # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
                               boxes_tensor = convert_coordinates(boxes_tensor, start_index=1, conversion='centroids2minmax')

                               # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
                               if (self$limit_boxes) {
                                 x_coords = boxes_tensor[, , , c(1, 2)]
                                 x_coords[x_coords >= self$img_width] = self$img_width - 1L
                                 x_coords[x_coords < 0] = 0L
                                 boxes_tensor[, , , c(1, 2)] = x_coords
                                 y_coords = boxes_tensor[, , , c(3, 4)]
                                 y_coords[y_coords >= self$img_height] = self$img_height - 1L
                                 y_coords[y_coords < 0] = 0L
                                 boxes_tensor[, , , c(3, 4)] = y_coords
                               }

                               # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
                               if (self$normalize_coords) {
                                 boxes_tensor[, , , 1:2] = boxes_tensor[, , , 1:2] / self$img_width
                                 boxes_tensor[, , , 3:4] = boxes_tensor[, , , 3:4] / self$img_height
                               }

                               if (self$coords == 'centroids') {
                                 # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
                                 # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
                                 boxes_tensor = convert_coordinates(boxes_tensor, start_index=1L, conversion='minmax2centroids')
                               }


                               # 4: Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
                               #    as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
                               # variances_tensor = array(rep(variances, each = feature_map_height * feature_map_width * self$n_boxes), dim = dim(boxes_tensor)) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
                               variances_tensor = np$zeros_like(boxes_tensor)
                               # variances_tensor += variances # Long live broadcasting
                               # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
                               # boxes_tensor = abind::abind(boxes_tensor, variances_tensor)
                               boxes_tensor = np$concatenate(reticulate::tuple(boxes_tensor, variances_tensor), axis = -1L)
                               # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
                               # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
                               # boxes_tensor = array(boxes_tensor, dim = c(1, dim(boxes_tensor)))
                               boxes_tensor = np$expand_dims(boxes_tensor, axis = 0L)
                               boxes_tensor = K$tile(K$constant(boxes_tensor, dtype='float32'), list(K$shape(x)[1], 1L, 1L, 1L, 1L))

                               return(boxes_tensor)
                             },

                             compute_output_shape = function(input_shape) {
                               if (K$image_dim_ordering() == 'tf') {
                                 batch_size = input_shape[[1]]
                                 feature_map_height = input_shape[[2]]
                                 feature_map_width = input_shape[[3]]
                                 feature_map_channels = input_shape[[4]]
                               } else {
                                 batch_size = input_shape[[1]]
                                 feature_map_channels = input_shape[[2]]
                                 feature_map_height = input_shape[[3]]
                                 feature_map_width = input_shape[[4]]
                               }
                               reticulate::tuple(batch_size, feature_map_height, feature_map_width, self$n_boxes, 8L)
                             }
                           )
)

layer_anchor <- function(object, img_height,
                         img_width,
                         this_scale,
                         next_scale,
                         aspect_ratios = c(0.5, 1.0, 2.0),
                         two_boxes_for_ar1 = TRUE,
                         limit_boxes = TRUE,
                         variances = c(1.0, 1.0, 1.0, 1.0),
                         coords = 'centroids',
                         normalize_coords = FALSE,
                         ...) {
  create_layer(AnchorBoxesLayer, object, list(
    img_height,
    img_width,
    this_scale,
    next_scale,
    aspect_ratios = c(0.5, 1.0, 2.0),
    two_boxes_for_ar1 = TRUE,
    limit_boxes = TRUE,
    variances = c(1.0, 1.0, 1.0, 1.0),
    coords = 'centroids',
    normalize_coords = FALSE,
    ...
  ))
}
