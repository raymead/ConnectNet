def get_pairs():
    tmp_pairs = []
    for i in range(6):
        for j in range(7):
            tmp_pairs.append([(i, j), (i+1,j), (i+2,j), (i+3,j)])
            tmp_pairs.append([(i, j), (i+1,j+1), (i+2,j+2), (i+3,j+3)])
            tmp_pairs.append([(i, j), (i,j+1), (i,j+2), (i,j+3)])
            tmp_pairs.append([(i, j), (i-1,j+1), (i-2,j+2), (i-3,j+3)])
    pairs_filt = []
    for plist in tmp_pairs:
        plist = [(x, y) for x, y in plist if 0 <= x < 6 and 0 <= y < 7]
        if len(plist) == 4:
            pairs_filt.append(([p[0] for p in plist], [p[1] for p in plist]))
    # # Checks
    # print(len(tmp_pairs))
    # print(len(pairs_filt))
    # print(3 * 7 + 4 * 6 + 3*4 + 3*4)
    return pairs_filt