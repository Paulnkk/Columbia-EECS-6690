# EECS-6690
We build a malware detection system for Android based applications with Machine Learning techniques (Random forest, Extra tree, Ada boost, XG boost, Gradient boost) in Python. Additionally, we explored new techniques (Deep Learning and SVM) to tackle the challenge of finding the best models to predict malicious software. Please find attached our paper with test results, descriptions and explanations.

from cvxpy.atoms.affine.reshape import reshape

def outer(x, y):

    if not (x.is_constant() or y.is_constant()):
        raise ValueError("At least one argument to outer must be constant.")
    elif any(vec.ndim > 1 for vec in [x, y]):
        raise ValueError("Outer requires vector/scalar arguments.")
    else:
        expr = reshape(x, (x.size, 1)) @ reshape(y, (y.size, 1))

    return expr
