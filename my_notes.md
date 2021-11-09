
# Option 1

1. Small linear regression model for a single user
2. Small individual linear regression model for top n users
3. Compare to global pairwise regression for top 20 users

## Small Model

## Many Models

One with not data transformation, one using normalized data.

- include users who have at least 150 ratings
- split each user's ratings into train/test with 80/20 split
- run k-fold cv in training data to find best predictive parameters
- use best models on test
- 

## Polynomial Model

# TODO:

- cleaned most of the duplicate tags
    - figure out which genres to use

- manually included several non-fiction tags
- run regression models on individual users
- run polynomial regression on multiple

- run many models
- compare to matrix factorization model
- consider rounding, or classification problem?
