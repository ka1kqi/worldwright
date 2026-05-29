# Risk-3 — IK+plan_path solvability confusion matrix

Total seeds run: 16  (dataset=`vs-risk3`, max_critic_iterations=2)

## Confusion matrix (solvable vs unsolvable)

| | pipeline=success | pipeline=fail |
|---|---|---|
| gt=solvable   | TP=4 | FN=2 |
| gt=unsolvable | FP=4 | TN=3 |

## Borderline (excluded from TP/FP/TN/FN)

- borderline_pass = 1
- borderline_fail = 2

## False-positive rate

FP / (FP + TN) = 4 / 7
false_positive_rate: 57.1%

## Followup

False-positive rate > 20%. See `sprints/sprint3_followup.md` for the proposed contact-stability post-grasp check.
