# EXP.NO.10-Simulation-of-Error-Detection-and-Correction-Algorithms
10.Simulation of Error Detecion and Correction Algorithms

# AIM

To implement a Linear Block Code (LBC) encoder and syndrome-based error detection and correction system using a generator matrix [G = I | P] and parity-check matrix [H = Pᵗ | I] in Python, and to analyze Hamming weight, minimum distance, and error correction capability.

# SOFTWARE REQUIRED

Google colab

# ALGORITHMS

1.Parity Check (Error Detection)

Count the number of 1’s in the data.

Add a parity bit to make the total number of 1’s even (even parity) or odd (odd parity).

At the receiver, recount the 1’s to detect any error.

2 Hamming Code (Error Detection and Correction)

Insert parity bits at positions that are powers of 2 (1, 2, 4, etc.).

Calculate each parity bit based on a specific set of data bits.

At the receiver, recompute parity to find the error position.

If an error is found, flip the bit to correct it.

# PROGRAM

``` python
import numpy as np

pb = []           # Parity matrix rows
Ik = []           # Identity matrix
p = []
m = []
h_dis = []
r_code = []
err = []

# Input for matrix dimensions
col = int(input("Enter the number of parity bits: "))
row = int(input("Enter the number of message bits: "))

# Input the parity matrix (P)
print("\nEnter the parity matrix rows:")
for i in range(row):
    p = list(map(int, input(f"Row {i+1}: ").split()))
    if len(p) != col:
        raise ValueError(f"Each row must have {col} elements.")
    pb.append(p)

# Convert to numpy array
p_mat = np.array(pb, dtype=int)
Ik = np.eye(row, dtype=int)

# Generator Matrix G = [I | P]
g_mat = np.hstack((Ik, p_mat))

# Codeword length n and message bits k
k = g_mat.shape[0]
n = g_mat.shape[1]

# Generate all possible message vectors (2^k)
all_messages = np.array([[1 if (i >> (k - j - 1)) & 1 else 0 for j in range(k)] for i in range(2 ** k)])

# Generate codewords: c = m * G mod 2
codewords = np.mod(np.dot(all_messages, g_mat), 2)

# Hamming weights
hamming_weights = [np.sum(row) for row in codewords]
d_min = np.min(hamming_weights[1:])

# Parity-check matrix H = [P^T | I]
p_t = p_mat.T
h_check = np.hstack((p_t, np.eye(col, dtype=int)))
ht = h_check.T  # Transpose of H

# Output Generator Matrix
print("\n**********")
print("Generator Matrix [G = I | P]:")
for row in g_mat:
    print("".join(map(str, row)))

# Output Codewords with Proper Formatting
print("\n**********")
print("{:<15} {:<15} {:<15}".format("Message Bits", "Codeword", "Hamming Weight"))
for i in range(len(all_messages)):
    msg_str = "".join(map(str, all_messages[i]))
    code_str = "".join(map(str, codewords[i]))
    weight = hamming_weights[i]
    print("{:<15} {:<15} {:<15}".format(msg_str, code_str, weight))

# Minimum Hamming distance
print("\n**********")
print(f"Minimum Hamming Distance: {d_min}")

# Output Parity Check Matrix
print("\n**********")
print("Parity Check Matrix [H = P^T | I]:")
for row in h_check:
    print("".join(map(str, row)))

# Output Transpose of Parity Check Matrix
print("\n**********")
print("Transpose of Parity Check Matrix (H^T):")
for row in ht:
    print("".join(map(str, row)))

# Receive codeword
rc = list(map(int, input("\nEnter the received codeword: ").split()))
if len(rc) != n:
    raise ValueError("Received codeword length must match codeword length n.")
r_c = np.array([rc])

# Syndrome calculation: s = r * H^T mod 2
syndrome = np.mod(np.dot(r_c, ht), 2).flatten()

# Find error position
err = np.zeros(n, dtype=int)
for i in range(n):
    if np.array_equal(syndrome, ht[i]):
        err[i] = 1
        break
print("\n**********")
print("Syndrome:", "".join(map(str, syndrome)))
print("Error vector:", "".join(map(str, err)))

# Correct the error
corrected = (r_c.flatten() + err) % 2
print("Corrected Codeword:", "".join(map(str, corrected)))

# Optional: Syndrome Table (first few entries)
print("\n**********")
print("Syndrome Matrix:")
for i in range(n):
    s = ht[i]
    ev = np.eye(n, dtype=int)[i]
    print(f"{' '.join(map(str, s))}  {' '.join(map(str, ev))}")

print("**********")
```


# OUTPUT

Enter the number of parity bits: 3
Enter the number of message bits: 3

Enter the parity matrix rows:
Row 1: 1 1 1
Row 2: 1 1 0
Row 3: 0 1 1

**********
Generator Matrix [G = I | P]:
1 0 0 1 1 1
0 1 0 1 1 0
0 0 1 0 1 1

**********
Message Bits	Codeword	Hamming Weight
0 0 0	0 0 0 0 0 0		0
0 0 1	0 0 1 0 1 1		3
0 1 0	0 1 0 1 1 0		3
0 1 1	0 1 1 1 0 1		4
1 0 0	1 0 0 1 1 1		4
1 0 1	1 0 1 1 0 0		3
1 1 0	1 1 0 0 0 1		3
1 1 1	1 1 1 0 1 0		4

**********
Minimum Hamming Distance: 3

**********
Parity Check Matrix [H = P^T | I]:
1 1 0 1 0 0
1 1 1 0 1 0
1 0 1 0 0 1

**********
Transpose of Parity Check Matrix (H^T):
1 1 1
1 1 0
0 1 1
1 0 0
0 1 0
0 0 1

Enter the received codeword: 1 0 1 1 0 1

**********
Syndrome: 0 0 1
Error vector: 0 0 0 0 0 1
Corrected Codeword: 1 0 1 1 0 0

**********
Syndrome Matrix:
1 1 1  1 0 0 0 0 0
1 1 0  0 1 0 0 0 0
0 1 1  0 0 1 0 0 0
1 0 0  0 0 0 1 0 0
0 1 0  0 0 0 0 1 0
0 0 1  0 0 0 0 0 1
**********

# RESULT / CONCLUSIONS

The code successfully generates all possible codewords using the Generator Matrix and calculates the minimum Hamming distance as 3, enabling single-bit error correction.
Syndrome decoding identifies and corrects errors in received codewords using the Parity Check Matrix.
