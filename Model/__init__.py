# Model package initialization.
#
# The two column models live in subpackages:
#   - Model/AlLDH : Al-LDH adsorption column (dimensional + non-dimensional)
#   - Model/IX    : ion-exchange column (Model/IX/dim and Model/IX/NonDim)
#
# Import the model you need directly from its subpackage, e.g.
#   from Model.IX.NonDim import qois, default_params