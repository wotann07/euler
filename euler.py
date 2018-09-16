import getopt
import itertools
import math
import sys
from decimal import Decimal
from math import log

__long_ass_digit = '73167176531330624919225119674426574742355349194934'


def get_dijkstra_shortest_path(weighted_graph, start, end):
    """
    Calculate the shortest path for a directed weighted graph.

    Node can be virtually any hashable datatype.

    :param start: starting node
    :param end: ending node
    :param weighted_graph: {"node1": {"node2": "weight", ...}, ...}
    :return: ["START", ... nodes between ..., "END"] or None, if there is no
             path
    """

    # We always need to visit the start
    nodes_to_visit = {start}
    visited_nodes = set()
    # Distance from start to start is 0
    distance_from_start = {start: 0}
    tentative_parents = {}

    while nodes_to_visit:
        # The next node should be the one with the smallest weight
        current = min([(distance_from_start[node], node) for node in nodes_to_visit])[
            1]  # [1] indicates we are taking the weight

        # The end was reached
        if current == end:
            break

        nodes_to_visit.discard(current)
        visited_nodes.add(current)

        edges = weighted_graph[current]
        unvisited_neighbours = set(edges).difference(visited_nodes)
        for neighbour in unvisited_neighbours:
            neighbour_distance = distance_from_start[current] + edges[neighbour]
            if neighbour_distance < distance_from_start.get(neighbour, float('inf')):
                distance_from_start[neighbour] = neighbour_distance
                tentative_parents[neighbour] = current
                nodes_to_visit.add(neighbour)

    return _deconstruct_path(tentative_parents, end)


def _deconstruct_path(tentative_parents, end):
    if end not in tentative_parents:
        return None
    cursor = end
    path = []
    while cursor:
        path.append(cursor)
        cursor = tentative_parents.get(cursor)
    return list(reversed(path))


def merge_sort(m):
    # base case
    if len(m) <= 1:
        return m

    # splitting
    half = len(m) // 2
    left = m[:half]
    right = m[half:]

    # recursive step
    left = merge_sort(left)
    right = merge_sort(right)

    # merging sorted
    return merge(left, right)


def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))

    while len(left) > 0:
        result.append(left.pop(0))
    while len(right) > 0:
        result.append(right.pop(0))

    print(result)
    return result


def quick_sort(m):
    quick_sort_helper(m, 0, len(m) - 1)
    print(m)


def quick_sort_helper(m, first, last):
    if first < last:
        p = partition(m, first, last)
        print(p)
        quick_sort_helper(m, first, p - 1)
        quick_sort_helper(m, p + 1, last)


def partition(m, first, last):
    print('first: %d - last: %d' % (first, last))
    pivot = m[last]
    i = first - 1
    for j in range(first, last):
        if m[j] < pivot:
            i += 1
            temp = m[i]
            m[i] = m[j]
            m[j] = temp

    if m[last] < m[i + 1]:
        temp = m[i + 1]
        m[i + 1] = m[last]
        m[last] = temp

    return i + 1


def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def is_abundant(n):
    return proper_divisors_sum(n) > n


def proper_divisors(n):
    i = 2
    divisors = [1]
    while i * i <= n:
        if not n % i:
            divisors.append(i)
            divisors.append(n / i)
        i += 1

    return divisors


def proper_divisors_sum(n):
    i = 2
    divisors_sum = 1
    while i * i <= n:
        if not n % i:
            divisors_sum += i + (n / i if math.sqrt(n) != i else 0)
        i += 1
    return divisors_sum


def max_product_grid(grid, n=4):
    max_product = 0
    for j in range(len(grid)):
        row = grid[j]
        for i in range(len(row)):
            if i <= len(row) - n:  # can always do forwards straight
                product = row[i]
                for p in range(1, n):
                    product *= row[i + p]
                max_product = product if product > max_product else max_product
                if j <= len(grid) - n:  # can also do forwards diagonal
                    product = row[i]
                    for p in range(1, n):
                        product *= grid[j + p][i + p]
                    max_product = product if product > max_product else max_product
            if i >= n - 1 and j <= len(grid) - n:  # can always do backward diagonal
                product = row[i]
                for p in range(1, n):
                    product *= grid[j + p][i - p]
                max_product = product if product > max_product else max_product
            if j <= len(grid) - n:  # can do straight down
                product = row[i]
                for p in range(1, n):
                    product *= grid[j + p][i]
                max_product = product if product > max_product else max_product

    return max_product


def is_criss_cross(grid):
    base_count = 0
    got_base = False
    d1_count = d2_count = 0
    d1_index = 0
    d2_index = len(grid[0]) - 1
    for y in range(len(grid)):
        row_count = 0
        column_count = 0
        d1_count += grid[y][d1_index]
        d1_index += 1
        d2_count += grid[y][d2_index]
        d2_index -= 1
        for x in range(len(grid[y])):
            if not got_base:
                base_count += grid[y][x]
            else:
                row_count += grid[y][x]

            column_count = 0
            for c in range(len(grid)):
                column_count += grid[c][x]

        if got_base:
            if row_count != base_count or column_count != base_count:
                return False
        else:
            got_base = True

    return d1_count == base_count and d2_count == base_count


def summatorial(n):
    return n * (n + 1) / 2


def pascal_row(n):
    row = [1]
    for k in range(1, n):
        row.append(row[k - 1] * (n + 1 - k) / k)
    if n is not 1:
        row.append(1)
    return row


def is_whole_integer(c):
    return c.real > 0 and Decimal(c.real) % 1 == 0 and c.imag == 0


def solve_quadratic(a, b, c):
    x = (-b + (b ** 2 - 4 * a * c) ** (1 / 2.0)) / 2
    y = (-b - (b ** 2 - 4 * a * c) ** (1 / 2.0)) / 2
    return x, y


def max_triangle(file_name):
    data = []
    with open(file_name) as f:
        for l in f.readlines():
            int_line = []
            for i in l.split():
                int_line.append(int(i))
            data.append(int_line)

    for i in range(len(data) - 2, -1, -1):
        for j in range(len(data[i])):
            data[i][j] = data[i][j] + \
                         max(data[i + 1][j], data[i + 1][j + 1])

    return data[0][0]


def collatz_sequence(n):
    # The Collazt sequence is defined by n % 2 -> n / 2 else 3 * n + 1
    seq = [n]
    while n != 1:
        if not n % 2:
            n /= 2
        else:
            n = 3 * n + 1
        seq.append(n)

    return seq


def triangle_number(n):
    return (n * (n + 1)) / 2


def factor_pairs(n):
    pairs = []
    for i in range(1, int(math.ceil(math.sqrt(n))) + 1):
        if not n % i:
            pairs.append((i, n / i))

    return pairs


def pythagorean_triplets(n):
    for i in range(2, n / 2, 2):
        for (s, t) in factor_pairs((i ** 2) / 2):
            print('factor pair: %s of %d' % ((s, t), i))
            a = i + s
            b = i + t
            c = i + t + s
            print(a, b, c)
            if (a + b + c) == n:
                return a * b * c


def biggest_adjacent_product(n=13):
    biggest = 0
    for i in range(len(__long_ass_digit) - n):
        digits = map(lambda c: int(c), __long_ass_digit[i: n + i])
        product = 1
        for j in digits:
            product *= j

        biggest = biggest if biggest > product else product

    return biggest


def is_prime(n):
    return len(prime_factorization(n)) == 1


def nth_prime(n=10001):
    i = 3
    count = 1

    if n == 1:
        return 2

    while True:
        if is_prime(i):
            count += 1

        if count == n:
            return i

        i += 2


def sum_square_difference(n=100):
    sq_sum = 0
    cum_sum = 0
    for i in range(1, n + 1):
        sq_sum += i ** 2
        cum_sum += i

    return cum_sum ** 2 - sq_sum


def is_palindrome(n):
    return str(n) == str(n)[::-1]


def largest_palindrome(n_digits=3):
    biggest = 0
    for i in range((10 ** n_digits - 1), (10 ** (n_digits - 1)) - 1, -1):
        for j in range((10 ** n_digits - 1), (10 ** (n_digits - 1)) - 1, -1):
            product = j * i
            if product > biggest and is_palindrome(product):
                biggest = product

    return biggest


def prime_factorization(n):
    i = 2
    primes = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            primes.append(i)

    if n > 1:
        primes.append(n)

    return primes


def largest_prime_factor(n=600851475143):
    return max(prime_factorization(n))


def fibonacci(n):
    if n == 0:
        raise ValueError('Invalid Argument: UNDEFINED for 0')
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibo_sum_even(upper=4000000):
    x = y = 1
    n = 0
    while n < upper:
        n += (x + y)
        x, y = x + 2 * y, 2 * x + 3 * y
        print(x, y)

    return n


def fibo_digit_count(digit_count=1000):
    x = y = 1
    i = 1
    while True:
        if len(str(x)) >= digit_count:
            print(x)
            return i
        elif len(str(y)) >= digit_count:
            print(y)
            return i + 1

        x, y = x + y, x + 2 * y
        i += 2


def three_five_multiples(below=1000):
    total = 0
    for i in range(below):
        if not i % 3 or not i % 5:
            total += i

    return total


def process(arg):
    if arg == '1':
        print(three_five_multiples())
    elif arg == '2':
        print(fibo_sum_even())
    elif arg == '3':
        print(largest_prime_factor())
    elif arg == '4':
        print(largest_palindrome())
    elif arg == '5':
        raise ValueError('Do it in your head')
    elif arg == '6':
        print(sum_square_difference())
    elif arg == '7':
        print(nth_prime())
    elif arg == '8':
        print(biggest_adjacent_product())
    elif arg == '9':
        print(pythagorean_triplets(1000))
    elif arg == '10':
        i = 3
        count = 2
        while i < 2000000:
            if is_prime(i):
                print(i)
                count += i

            i += 2

        print(count)
    elif arg == '11':
        grid = []
        with open('p11.txt') as f:
            for l in f.readlines():
                grid.append([int(x) for x in l.split()])

        print(max_product_grid(grid))

    elif arg == '12':
        i = 10
        while True:
            value = triangle_number(i)
            divisors = factor_pairs(value)
            if len(divisors) > 250:
                print(value)
                break
            i += 1
    elif arg == '13':
        with open('p13.txt') as f:
            addition = 0
            for l in f.readlines():
                addition += int(l)

            print(str(addition)[:10])
    elif arg == '14':
        longest = 0
        starter = 0
        for i in range(1, 1000000):
            seq = collatz_sequence(i)
            (longest, starter) = (longest, starter) if len(
                seq) < longest else (len(seq), i)
        print(longest, starter)
    elif arg == '15':
        print(binomial_coefficient(40, 20))
    elif arg == '16':
        print(sum(int(c) for c in str(2 ** 1000)))
    elif arg == '18':
        print(max_triangle('p18.txt'))
    elif arg == '20':
        n = math.factorial(100)
        count = 0
        for c in str(n):
            count += int(c)
        print(count)
    elif arg == '21':
        amicable_dict = {}
        result = 0
        for i in range(2, 10000):
            amicable_dict[i] = proper_divisors_sum(i)
        for key in amicable_dict.keys():
            if amicable_dict[key] in amicable_dict and key != amicable_dict[key] and \
                    amicable_dict[amicable_dict[key]] == key:
                result += key
        print(result)
    elif arg == '22':
        names = []
        with open('p22.txt') as f:
            for l in f.readlines():
                names = map(lambda s: s.strip('"'), l.split(','))
        names.sort()
        result = 0
        for i in range(len(names)):
            name_weight = 0
            for c in names[i]:
                name_weight += ord(c) - 64
            result += name_weight * (i + 1)
        print(result)
    elif arg == '23':
        abundant = []
        for i in range(2, 28123):
            if is_abundant(i):
                abundant.append(i)
        print(abundant)
    elif arg == '25':
        print(fibo_digit_count())
    elif arg == '67':
        print(max_triangle('p67.txt'))
    elif arg == '99':
        data = []
        with open('p99.txt') as f:
            data = list(map(lambda s: (int(s[0]), int(s[1])), map(lambda line: line.split(','), f.readlines())))
        print(max(range(len(data)), key=lambda p: data[p][1] * log(data[p][0])) + 1)
    elif arg == '100':
        n = 10 ** 12 - 1
        while True:
            (x, y) = solve_quadratic(1, -1, -(n * (n - 1) / 2))
            print(x.real, n, y.real, x, y)
            if is_whole_integer(x):
                break
            elif is_whole_integer(y):
                break
            n = n + 1

        print(x, y)
    elif arg == '148':
        c = 0
        print(pascal_row(2))
        for i in range(1, 21):
            row = pascal_row(i)
            print(row)
            for j in row:
                if not 0 == j % 7:
                    c += 1
        print(c)
        # total of numbers in 10^9
        n = 14
        entities = summatorial(n)
        print(entities)
        # rows that are divisible are the sum of the entities on rows multiple of 7 minus 2 * number of rows
        divisible_number_of_rows = (n - n % 7) / 7
        print(divisible_number_of_rows)
        divisible_entities = summatorial(
            divisible_number_of_rows) * 7 - divisible_number_of_rows  # accounts for the extra 1
        undivisible = entities - divisible_entities
        print(undivisible)
    elif arg == '166':
        a = list(itertools.product(range(10), repeat=4))
        count = 0
        for x_0 in range(len(a)):
            for x_1 in range(len(a)):
                for x_2 in range(len(a)):
                    for x_3 in range(len(a)):
                        if is_criss_cross((a[x_0], a[x_1], a[x_2], a[x_3])):
                            count += 1
        print(count)

    else:
        raise ValueError('Not Implemented')


def length_of_longest_substring(s):
    """
    :type s: str
    :rtype: int
    """
    holder = ''
    longest = ''
    sub_start = 0
    char_pos = 1
    for c in s:
        if c not in holder:
            holder = s[sub_start:char_pos]
        else:
            if len(holder) > len(longest):
                longest = holder
            holder = c
            sub_start += 1
        char_pos += 1

    return len(longest) if len(longest) > len(holder) else len(holder)


def is_rotated(arr, rot):
    # find start of arr in rot
    # open iteration through rot through modulo
    """
        if len(arr) != len(rot):
        return False

    i = 0
    j = 0
    found_start = False
    for e in rot:
        if e == arr[0]:
            found_start = True
            j = i
            for other_e in arr:
                if rot[j % len(rot)] != other_e:
                    found_start = False
                    break
                j += 1
        if found_start:
            break
        i += 1

    return found_start
    """

    new_arr = rot + rot
    return any(arr == new_arr[offset:offset + len(arr)] for offset in range(len(new_arr)))


def cell_compete(states, days):
    # WRITE YOUR CODE HERE
    if len(states) < 1:
        raise Exception('Empty array')
    elif len(states) is 1:
        return 0
    else:
        new_state = states[:]
        for n in range(days):
            new_state[0] = 0 ^ states[1]
            new_state[len(new_state) - 1] = 0 ^ states[len(states) - 2]
            for i in range(1, len(states) - 1):
                new_state[i] = states[i - 1] ^ states[i + 1]
            states = new_state[:]
        return new_state


def __find_gcd(a, b):
    return b if a == 0 else __find_gcd(b % a, a)


def __find_gcd_alt(a, b):
    while b:
        a, b = b, b % a
    return a


def generalized_gcd(num, arr):
    # WRITE YOUR CODE HERE
    # using Euclidean algorithm
    gcd = arr[0]
    for n in arr:
        print gcd
        gcd = __find_gcd(n, gcd)

    return gcd


def main():
    print(cell_compete([1, 0, 1, 1, 0, 0, 1], 2))
    print(is_rotated([1, 2, 3, 4], [3, 4, 1, 2]))
    print(is_rotated([1, 2, 3, 4], [3, 5, 1, 2]))
    print(is_rotated([1, 2, 3, 4, 1], [3, 4, 1, 1, 2]))
    print(is_rotated([1, 2, 2, 2, 23, 4], [23, 4, 1, 2, 2, 2]))
    print(length_of_longest_substring('dvdf'))
    print(quick_sort([2, 5, 1, 43, 22, 50, 3, 4, 5]))
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    except getopt.error as msg:
        print(msg)
        print('for help use --help')
        sys.exit(2)

    # process options
    for o in opts:
        if o in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)

    # process arguments
    euler = input("Please enter problem number: ")
    process(euler.strip())


if __name__ == '__main__':
    sys.exit(main())
