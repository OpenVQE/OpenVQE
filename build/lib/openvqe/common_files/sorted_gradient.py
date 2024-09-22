#!/usr/bin/env python
# coding: utf-8


def corresponding_index(new_list, new_list_index, sorted_new):
    """
    It generates sorted index after shuffling of elements in list.
    """
    n = []
    res = []
    for i in range(0, len(sorted_new)):
        for j in range(0, len(new_list)):
            if new_list[j] == sorted_new[i]:
                n.append(new_list_index[j])
            else:
                pass
    for i in n:
        if i not in res:
            res.append(i)
    return res


def index_without_0(my_list):
    """
    index_without_0 generate a list of index without 0.

    it takes list with 0 and returns list of index without 0

    """
    u = []
    for i in range(0, len(my_list)):
        if my_list[i] != 0:
            u.append(i)
    return u


def value_without_0(my_list):
    """
    Value_without_0 will remove 0 from the list.

    parameters
    -----------
    my_list: List<float>
        input list
    
    Returns
    --------
    new: List<float>
        the list without zeros
    """
    new = []
    for i in range(0, len(my_list)):
        if my_list[i] != 0:
            new.append(my_list[i])
    return new


def apply_neg(sorted_list, neg_num, occ_dict):
    """Internal function for internal working of program"""
    n = []
    for i in sorted_list:
        if (i * (-1) in neg_num) and (-i not in n):
            n.append(-i)
            occ = occ_dict[i * (-1)]
            index = sorted_list.index(i)
            for j in range(occ):
                sorted_list[index] *= -1
                index += 1
    return sorted_list


def abs_sort_desc(my_list) -> list:
    """Internal function for internal working of program"""
    occ_dict, occ_list = occurence(my_list)
    ext_neg_num = []

    # make list positive
    for i in range(0, len(my_list)):
        if my_list[i] < 0:
            ext_neg_num.append(my_list[i])
            my_list[i] *= -1

    my_list.sort()
    my_list.reverse()

    # add negative number in list which extracted in ext_neg_num
    my_list = apply_neg(my_list, ext_neg_num, occ_dict)
    return my_list


def occurence(my_list) -> dict:
    """
    occurence function gives the counts of particular value in list.

    Parameter of this list is 'mylistt' and returns dictionary of occurence

    """
    n = []
    nn = []
    my_dict = {i: my_list.count(i) for i in my_list}
    for key, item in my_dict.items():
        for k, i in my_dict.items():
            if key == k * (-1):
                n.append(key)
    for i in range(0, len(n)):
        if n[i] < 0:
            nn.append(n[i])
    return my_dict, nn


def duplicates(my_list, item):
    return [i for i, x in enumerate(my_list) if x == item]
