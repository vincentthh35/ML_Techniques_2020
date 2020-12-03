# calculate number of weights
def my_func(my_list):
    l = len(my_list)
    ret = 12 * (my_list[0] - 1)
    for i in range(1, l):
        ret += my_list[i - 1] * (my_list[i] - 1)
    ret += my_list[l - 1]
    return ret

def recursion(my_list, available, current_layer, max_layer):
    if current_layer == max_layer - 1:
        # the last element
        my_list.append(available)
        global max_list, my_max
        # calculate number of weights
        # print(my_list)
        ret = my_func(my_list)
        if ret > my_max:
            max_list = []
            for element in my_list:
                max_list.append(element)
            my_max = ret
        my_list.pop()
        return
    elif available <= 0:
        return
    for i in range(2, available - (max_layer - current_layer - 1) * 2 + 1):
        my_list.append(i)
        recursion(my_list, available - i, current_layer + 1, max_layer)
        my_list.pop()
    return

max_hidden_L = 24
max_hidden_neuron = 48
my_max = 0
max_list = []
for l in range(1, max_hidden_L + 1):
    recursion([], max_hidden_neuron, 0, l)
    print(f'***** iteration: l = {l}')
    print(f'current : my_max = {my_max}, max_list = {max_list}\n')

print(f'my_max = {my_max}, max_list = {max_list}')
