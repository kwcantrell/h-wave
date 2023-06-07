track = [0, 0, 0, 0, 0, 0]
postorder = []
max_children=2
while track[0] < 1:
    i=len(track)-1
    while True:
        postorder.append((i, track[i]))
        track[i] += 1
        if (track[-1]) % max_children == 0:
            break
    
    i -= 1
    postorder.append((i, track[i]))
    track[i] += 1
    while (track[i]) % max_children == 0 and i >= 0:
        i -= 1
        postorder.append((i, track[i]))
        track[i] += 1

def get_gap(level):    
    gaps = []
    cur_gap = 0
    for (i, j ) in postorder:
        if i == len(track)-l:
            gaps.append(cur_gap)
            cur_gap = 0
        cur_gap += 1
    return gaps    
for l in range(1, len(track), 1):
    print(get_gap(l), (max_children**l-max_children)/(max_children-1))
