import numpy as np
from numpy.linalg import norm

z_org = np.load("z_org.npy").flatten()
z_low = np.load("z_low.npy").flatten()
z_high = np.load("z_high.npy").flatten()

org_low = np.dot(z_org, z_low) / (norm(z_org)*norm(z_low))
org_high = np.dot(z_org, z_high) / (norm(z_org)*norm(z_high))
low_high = np.dot(z_high, z_low) / (norm(z_high)*norm(z_low))
test = np.dot(z_org, z_org) / (norm(z_org)*norm(z_org))

with open('latent_dim_comparison', 'w') as f:
    f.write(f"org_low: {org_low}\n")
    f.write(f"org_high: {org_high}\n")
    f.write(f"low_high: {low_high}\n")
    f.write(f"test: {test}\n")