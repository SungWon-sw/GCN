def get_size(n, edge, check, start):
    s = [(start, 0, False)]; size = [0] * n
    while s:
        node, prev, visited = s.pop()

        if visited:
            size[prev] += size[node]
        else:
            size[node] = 1; s.append((node, prev, True))
            for next in edge[node]:
                if prev == next or check[next]: continue
                s.append((next, node, False))

    return size

def get_centroid(edge, size, check, start):
    s = [(start, 0)]; cent = 0
    while s:
        node, prev = s.pop()
        
        for next in edge[node]:
            if prev == next or check[next]: continue
            if size[next] * 2 > size[start]: s.append((next, node))

        if not s: cent = node; break

    return cent

def get_centroid_tree(n, edge):
    s = [(0, 0)]; check = [0] * n; cent_tree = []
    while s:
        node, prev = s.pop()
        size = get_size(n, edge, check, node)
        cent = get_centroid(edge, size, check, node)

        if prev: cent_tree.append((prev, cent))

        check[cent] = True
        for next in edge[cent]:
            if check[next]: continue

            s.append((next, cent))     

    return cent_tree

def get_vn_tree(n, edge):
    return get_centroid_tree(n, edge)

def get_vn_connect(vn, bag):
    vn_connect = []
    for i in vn:
        for node in bag[i]:
            vn_connect.append((i, node))
    return vn_connect

def op(n, edge):
    bags, bag_tree = tree_decomposition(n, edges)
    
    m = len(bags)
    vn_tree = get_vn_tree(m, bag_tree)

    vn = set()
    
    for vn_edge in vn_tree:
        vn_edge[0] += n
        vn_edge[1] += n
        vn.add(vc_edge[0])
        vn.add(vc_edge[1])
        
    new_tree = edge + vn_tree + get_vn_connect(vn, bags)
    
    return (n+m, new_tree)
