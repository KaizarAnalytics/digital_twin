import matplotlib.pyplot as plt
def hist_max_occ(max_occ, capacity, title=None):
    plt.figure(figsize=(8,5))
    plt.hist(max_occ, bins=30, edgecolor="black")
    plt.axvline(capacity, linestyle="--", label=f"Capacity={capacity}")
    if title: plt.title(title)
    plt.xlabel("Max occupancy (over horizon)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
