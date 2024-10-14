import numpy as np

def calculate_joint_probability(byte_array):
    total_count = np.sum(byte_array)

    # P(X)
    P_X = byte_array / total_count

    # Calculate the entropy H(X) using Shannon entropy formula
    P_X_nonzero = P_X[P_X > 0]

    # H(X) 엔트로피
    H_X = -np.sum(P_X_nonzero * np.log2(P_X_nonzero))

    # Calculate the joint probability P(H, X) as P(X) * P(H|X)
    # Here we assume P(H|X) is proportional to H(X) for simplicity.
    # P(H|X) would typically be derived from a conditional distribution, but we use this approach here.
    P_H_given_X = P_X * H_X  # Simplified model for illustration
    print(P_X.shape)
    print(H_X)

    P_H_X = P_X * P_H_given_X

    return P_H_X

# Example byte array (256 possible byte values with different counts)
byte_array = np.random.randint(0, 100, size=256)

# Calculate joint probability P(H, X)
joint_probabilities = calculate_joint_probability(byte_array)
print(joint_probabilities.shape)
