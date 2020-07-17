package neuron

import (
	"fmt"
	"math"
)


func meanAbsoluteError(predictions, targets []float64) (float64, error) {
	if len(predictions) != len(targets) {
		return 0, fmt.Errorf("length of predictions does not match length of targets")
	}

	errAcc := float64(0)

	for i, p := range predictions {
		errAcc += math.Abs(p - targets[i])
	}

	return (1.0 / float64(len(predictions))) * errAcc, nil
}


func meanSquaredError(predictions, targets []float64) (float64, error) {
	if len(predictions) != len(targets) {
		return 0, fmt.Errorf("length of predictions does not match length of targets")
	}

	errAcc := float64(0)

	for i, p := range predictions {
		errAcc += math.Pow(p - targets[i], 2)
	}

	return (1.0 / float64(2 * len(predictions))) * errAcc, nil
}