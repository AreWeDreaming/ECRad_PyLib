import numpy as np
def bubble_sort_two(indices, second_array):
    while True:
        nothing_sorted = True
        for i in range(len(indices) - 1):
            if(indices[i] > indices[i + 1]):
                temp = indices[i]
                indices[i] = indices[i + 1]
                indices[i + 1] = temp
                temp = second_array[i]
                second_array[i] = second_array[i + 1]
                second_array[i + 1] = temp
                nothing_sorted = False
        if(nothing_sorted):
            break

def test_bubble_sort():
    data = np.random.rand(50)
    indices = np.linspace(1, 100, 50, dtype=np.int32)[::-1]
    print(indices)
    bubble_sort_two(indices, data)
    print(indices)
test_bubble_sort()
