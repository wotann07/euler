import getopt
import math
import sys
from decimal import Decimal
from math import log

__long_ass_digit = '73167176531330624919225119674426574742355349194934'


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
            print 'factor pair: %s of %d' % ((s, t), i)
            a = i + s
            b = i + t
            c = i + t + s
            print (a, b, c)
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
        print (x, y)

    return n


def three_five_multiples(below=1000):
    total = 0
    for i in range(below):
        if not i % 3 or not i % 5:
            total += i

    return total


def process(arg):
    if arg == '1':
        print three_five_multiples()
    elif arg == '2':
        print fibo_sum_even()
    elif arg == '3':
        print largest_prime_factor()
    elif arg == '4':
        print largest_palindrome()
    elif arg == '5':
        raise ValueError('Do it in your head')
    elif arg == '6':
        print sum_square_difference()
    elif arg == '7':
        print nth_prime()
    elif arg == '8':
        print biggest_adjacent_product()
    elif arg == '9':
        print pythagorean_triplets(1000)
    elif arg == '10':
        i = 3
        count = 2
        while i < 2000000:
            if is_prime(i):
                print i
                count += i

            i += 2

        print count
    elif arg == '12':
        i = 10
        while True:
            value = triangle_number(i)
            divisors = factor_pairs(value)
            if len(divisors) > 250:
                print value
                break
            i += 1
    elif arg == '13':
        with open('p13.txt') as f:
            addition = 0
            for l in f.readlines():
                addition += int(l)

            print str(addition)[:10]
    elif arg == '14':
        longest = 0
        starter = 0
        for i in range(1, 1000000):
            seq = collatz_sequence(i)
            (longest, starter) = (longest, starter) if len(
                seq) < longest else (len(seq), i)
        print (longest, starter)
    elif arg == '15':
        addition = 0
        power = str(2 ** 1000)
        print power
        for c in power:
            addition += int(c)
        print addition
    elif arg == '18':
        print max_triangle('p18.txt')
    elif arg == '20':
        n = math.factorial(100)
        count = 0
        for c in str(n):
            count += int(c)
        print count
    elif arg == '67':
        print max_triangle('p67.txt')
    elif arg == '99':
        data = []
        with open('p99.txt') as f:
            data = map(lambda s: (int(s[0]), int(s[1])), map(lambda line: line.split(','), f.readlines()))
        print max(range(len(data)), key=lambda p: data[p][1] * log(data[p][0])) + 1
    elif arg == '100':
        n = 10 ** 12 - 1
        while True:
            (x, y) = solve_quadratic(1, -1, -(n * (n - 1) / 2))
            print (x.real, n, y.real, x, y)
            if is_whole_integer(x):
                break
            elif is_whole_integer(y):
                break
            n = n + 1

        print (x, y)
    elif arg == '148':
        c = 0
        print pascal_row(2)
        for i in range(1, 21):
            row = pascal_row(i)
            print row
            for j in row:
                if not 0 == j % 7:
                    c += 1
        print c
        # total of numbers in 10^9
        n = 14
        entities = summatorial(n)
        print entities
        # rows that are divisible are the sum of the entities on rows multiple of 7 minus 2 * number of rows
        divisible_number_of_rows = (n - n % 7) / 7
        print divisible_number_of_rows
        divisible_entities = summatorial(
            divisible_number_of_rows) * 7 - divisible_number_of_rows  # accounts for the extra 1
        undivisible = entities - divisible_entities
        print undivisible
    else:
        raise ValueError('Not Implemented')


def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    except getopt.error, msg:
        print msg
        print 'for help use --help'
        sys.exit(2)

    # process options
    for o in opts:
        if o in ('-h', '--help'):
            print __doc__
            sys.exit(0)

    # process arguments
    while True:
        euler = raw_input("Please enter problem number: ")
        process(euler.strip())


if __name__ == '__main__':
    sys.exit(main())
