test = [1,2,3,4,5,6,7,8]

def search(seq, target: int):
    """
    Given a sorted list of integers, find the index of a target if
    it exists in the list. If not, return the next smallest integer.
    """
    lo = 0
    hi = len(seq) - 1
    print(f"lo: {lo}, hi: {hi}")
    while lo < hi:
        mid = (lo + hi) // 2
        if target < seq[mid]:
            hi = mid
        else:
            lo = mid + 1
        print(f"arr[{lo}]={seq[lo]} arr[{hi}]={seq[hi]}")
    
    return lo
    


# for i in [1,4,5,4.5,10]:
#     print(f"target: {i}")
#     print(search(test, i))

print(search(test, 1.5))

    