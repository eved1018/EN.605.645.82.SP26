M = [[0, 0, 0, 0], [0, -1, -1, 0], [0, 0, 0, 0], [0, 0, -1, 1]]


def enqueue(q, elem):
    q.append(elem)


def dequeue(q):
    return q.pop(0)


def push(stack, elem):
    stack.append(elem)


def pop(stack):
    return stack.pop()


def neighbors(G, S):
    ns = []
    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        x = dx + S[0]
        y = dy + S[1]
        if 0 <= x <= 3 and 0 <= y <= 3 and G[y][x] != -1:
            ns.append((x, y))
    return ns


def g2s(G, S, stack ):
    s = ""
    for y, row in enumerate(G):
        for x, val in enumerate(row):
            if (x,y) == S:
                s += " * "
            elif (x,y) in stack:
                s += " f "
            elif val == 0:
                s += " o "
            elif val == -1:
                s += " x "
            elif val == 1:
                s += " G "
        s += "\n"
    return s

def dfs(G, S):
    stack = [S]
    visited = set()
    path = []
    prev = S

    while stack:
        node = stack.pop()

        if G[node[1]][node[0]] == 1:
            path.append(node)
            return path

        visited.add(node)
        path.append(node)

        for n in neighbors(G, node):
            if n not in visited:
                stack.append(n)

        print(f"T: {prev} -> {node} | F: {stack} | E: {visited}")
        print(g2s(G, node, stack))
        prev = node



def bfs(G, S):
    q = [S]
    visited = set()
    path = []
    prev = S

    while q:
        node = dequeue(q)
        print(f"{prev} -> {node}")
        prev = node

        x,y = node  
        if G[y][x] == 1:
            path.append(node)
            return path

        if node not in visited:
            visited.add(node)
            path.append(node)

            for n in neighbors(G, node):
                if n not in visited:
                    enqueue(q, n)



path = dfs(M, (0, 0))
print(path)

path = bfs(M, (0, 0))
print(path)

