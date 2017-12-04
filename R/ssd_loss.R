#' @export
SSDLoss <- R6::R6Class("SSDLoss",
                       public = list(
                         neg_pos_ratio = 3L,
                         n_neg_min = 0L,
                         alpha = 1.0,
                         n_classes = NULL,
                         tf = tensorflow::tf,

                         initialize = function(neg_pos_ratio = 3L,
                                               n_neg_min = 0L,
                                               alpha = 1.0,
                                               n_classes = NULL) {
                           self$neg_pos_ratio = self$tf$constant(neg_pos_ratio)
                           self$n_neg_min = self$tf$constant(n_neg_min)
                           self$alpha = self$tf$constant(alpha)
                           self$n_classes = as.integer(n_classes)
                         },

                         smooth_L1_loss = function(y_true, y_pred) {
                           y_true = self$tf$cast(y_true, "float32")
                           absolute_loss = self$tf$abs(y_true - y_pred)
                           square_loss = 0.5 * (y_true - y_pred)**2
                           l1_loss = self$tf$where(self$tf$less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
                           return(self$tf$reduce_sum(l1_loss, axis = -1L))
                         },

                         log_loss = function(y_true, y_pred) {
                           y_true = self$tf$cast(y_true, "float32")
                           # Make sure that `y_pred` doesn't contain any zeros (which would break the log function)
                           y_pred = self$tf$maximum(y_pred, 1e-15)
                           # Compute the log loss
                           log_loss = -self$tf$reduce_sum(y_true * self$tf$log(y_pred), axis = -1L)
                           return(log_loss)
                         },
                         compute_loss = function(y_true, y_pred) {
                           y_true$set_shape(y_pred$get_shape())
                           batch_size = self$tf$shape(y_pred)[1] # Output dtype: tf.int32
                           n_boxes = self$tf$shape(y_pred)[2] # Output dtype: tf.int32, note that `n_boxes` in this context denotes the total number of boxes per image, not the number of boxes per cell

                           # 1: Compute the losses for class and box predictions for every box
                           classification_loss = self$tf$to_float(self$log_loss(y_true[,,1:self$n_classes], y_pred[,,1:self$n_classes])) # Output shape: (batch_size, n_boxes)
                           localization_loss = self$tf$to_float(self$smooth_L1_loss(y_true[,,(self$n_classes + 1):(self$n_classes + 4)], y_pred[,,(self$n_classes + 1):(self$n_classes + 4)])) # Output shape: (batch_size, n_boxes)

                           # 2: Compute the classification losses for the positive and negative targets

                           # Create masks for the positive and negative ground truth classes
                           negatives = y_true[,,1] # Tensor of shape (batch_size, n_boxes)
                           positives = self$tf$to_float(self$tf$reduce_max(y_true[,,2:self$n_classes], axis=-1L)) # Tensor of shape (batch_size, n_boxes)

                           # Count the number of positive boxes (classes 1 to n) in y_true across the whole batch
                           n_positive = self$tf$reduce_sum(positives)
                           # Now mask all negative boxes and sum up the losses for the positive boxes PER batch item
                           # (Keras loss functions must output one scalar loss value PER batch item, rather than just
                           # one scalar for the entire batch, that's why we're not summing across all axes)
                           pos_class_loss = self$tf$reduce_sum(classification_loss * positives, axis=-1L) # Tensor of shape (batch_size,)

                           # Compute the classification loss for the negative default boxes (if there are any)

                           # First, compute the classification loss for all negative boxes
                           neg_class_loss_all = classification_loss * negatives # Tensor of shape (batch_size, n_boxes)
                           n_neg_losses = self$tf$count_nonzero(neg_class_loss_all, dtype=self$tf$int32) # The number of non-zero loss entries in `neg_class_loss_all`

                           # What's the point of `n_neg_losses`? For the next step, which will be to compute which negative boxes enter the classification
                           # loss, we don't just want to know how many negative ground truth boxes there are, but for how many of those there actually is
                           # a positive (i.e. non-zero) loss. This is necessary because `tf.nn.top-k()` in the function below will pick the top k boxes with
                           # the highest losses no matter what, even if it receives a vector where all losses are zero. In the unlikely event that all negative
                           # classification losses ARE actually zero though, this behavior might lead to `tf.nn.top-k()` returning the indices of positive
                           # boxes, leading to an incorrect negative classification loss computation, and hence an incorrect overall loss computation.
                           # We therefore need to make sure that `n_negative_keep`, which assumes the role of the `k` argument in `tf.nn.top-k()`,
                           # is at most the number of negative boxes for which there is a positive classification loss.

                           # Compute the number of negative examples we want to account for in the loss
                           # We'll keep at most `self.neg_pos_ratio` times the number of positives in `y_true`, but at least `self.n_neg_min` (unless `n_neg_loses` is smaller)
                           n_negative_keep = self$tf$minimum(self$tf$maximum(self$neg_pos_ratio * self$tf$to_int32(n_positive), self$n_neg_min), n_neg_losses)

                           # In the unlikely case when either (1) there are no negative ground truth boxes at all
                           # or (2) the classification loss for all negative boxes is zero, return zero as the `neg_class_loss`
                           f1 = function() {
                             return(self$tf$zeros(list(batch_size)))
                           }
                           # Otherwise compute the negative loss
                           f2 = function() {
                             # Now we'll identify the top-k (where k == `n_negative_keep`) boxes with the highest confidence loss that
                             # belong to the background class in the ground truth data. Note that this doesn't necessarily mean that the model
                             # predicted the wrong class for those boxes, it just means that the loss for those boxes is the highest.

                             # To do this, we reshape `neg_class_loss_all` to 1D...
                             neg_class_loss_all_1D = self$tf$reshape(neg_class_loss_all, list(-1L)) # Tensor of shape (batch_size * n_boxes,)
                             # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
                             topKRes = self$tf$nn$top_k(neg_class_loss_all_1D, n_negative_keep, FALSE) # We don't need sorting
                             values = topKRes$values
                             indices = topKRes$indices

                             # ...and with these indices we'll create a mask...
                             negatives_keep = self$tf$scatter_nd(self$tf$expand_dims(indices, axis=1L), updates=self$tf$ones_like(indices, dtype=self$tf$int32), shape=self$tf$shape(neg_class_loss_all_1D)) # Tensor of shape (batch_size * n_boxes,)
                             negatives_keep = self$tf$to_float(self$tf$reshape(negatives_keep, list(batch_size, n_boxes)))

                             # ...and use it to keep only those boxes and mask all other classification losses
                             neg_class_loss = self$tf$reduce_sum(classification_loss * negatives_keep, axis=-1L) # Tensor of shape (batch_size,)

                             return(neg_class_loss)
                           }

                           neg_class_loss = self$tf$cond(self$tf$equal(n_neg_losses, self$tf$constant(0L)), f1, f2)

                           class_loss = pos_class_loss + neg_class_loss # Tensor of shape (batch_size,)

                           # 3: Compute the localization loss for the positive targets
                           #    We don't penalize localization loss for negative predicted boxes (obviously: there are no ground truth boxes they would correspond to)

                           loc_loss = self$tf$reduce_sum(localization_loss * positives, axis=-1L) # Tensor of shape (batch_size,)

                           # 4: Compute the total loss

                           total_loss = (class_loss + self$alpha * loc_loss) / self$tf$maximum(1.0, n_positive) # In case `n_positive == 0`

                           return(total_loss)
                         }
                       )
)
