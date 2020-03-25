def convertHierarchyDict2labelSplitters(hierarchy_dict):
    splitters = []
    foo = []
    for d in hierarchy_dict.values():
        foo = d
        splitters.append(d.index(1))
    splitters.append(len(foo))
    return splitters
# convertHierarchyDict2labelSplitters(hierarchy_dict['1'])

def convertClass2HierarchicalClass(train_labels, hierarchical_class_list):
    new_class_list = []
    for l in train_labels:
#         print('next')
#         print(hierarchical_class_list)
        for i, hl in enumerate(hierarchical_class_list):
            if int(l) < hl:
                new_class_list.append(i-1)
                break
    return new_class_list

def convertClassWeights2HierarchicalClassWeights(class_weights, hierarchical_class_list):
    new_class_weights = []
    count = 0
#     print(hierarchical_class_list)
    for i in range(hierarchical_class_list[-1]):
#         print(new_class_weights)
        for hl in hierarchical_class_list[1:]:
#             print(i, hl)
            if i < hl:
#                 print(class_weights)
                new_class_weights.append(class_weights[count])
                break
            if i == hl:
                count += 1
                new_class_weights.append(class_weights[count])
                break
    print(new_class_weights)
    return new_class_weights