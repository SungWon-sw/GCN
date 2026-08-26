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
    s = [(0, -1)]; check = [0] * n; cent_tree = []
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

def get_vn_connect(n, m, bag):
    vn_connect = []
    for i in range(n, n+m):
        for node in bag[i - n]:
            vn_connect.append((i, node))
    return vn_connect

def op(n, edge):
    bags, bag_tree = tree_decomposition(n, edge)
    
    m = len(bags)
    adj_bag_tree = [[] for _ in range(m)]
    
    for u, v in bag_tree:
        adj_bag_tree[u].append(v)
        adj_bag_tree[v].append(u)
    
    base_vn_tree = get_vn_tree(m, adj_bag_tree)
    vn_edge = [(u + n, v + n) for u, v in base_vn_tree]
        
    new_tree = edge + vn_edge + get_vn_connect(n, m, bags)
    
    return (n+m, new_tree)
