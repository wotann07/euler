import getopt
import math
import sys

__long_ass_digit = '73167176531330624919225119674426574742355349194934\
96983520312774506326239578318016984801869478851843\
85861560789112949495459501737958331952853208805511\
12540698747158523863050715693290963295227443043557\
66896648950445244523161731856403098711121722383113\
62229893423380308135336276614282806444486645238749\
30358907296290491560440772390713810515859307960866\
70172427121883998797908792274921901699720888093776\
65727333001053367881220235421809751254540594752243\
52584907711670556013604839586446706324415722155397\
53697817977846174064955149290862569321978468622482\
83972241375657056057490261407972968652414535100474\
82166370484403199890008895243450658541227588666881\
16427171479924442928230863465674813919123162824586\
17866458359124566529476545682848912883142607690042\
24219022671055626321111109370544217506941658960408\
07198403850962455444362981230987879927244284909188\
84580156166097919133875499200524063689912560717606\
05886116467109405077541002256983155200055935729725\
71636269561882670428252483600823257530420752963450'


def collatz_sequence(n):
    return


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
    n = 0
    i = 2
    total = 0
    while True:
        n = fibonacci(i)
        if n < upper:
            i += 3
            total += n
        else:
            break

    return total


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
        i = 10
        while True:
            value = triangle_number(i)
            divisors = factor_pairs(value)
            if len(divisors) > 250:
                print(value)
                break
            i += 1
    elif arg == '13':
        with open('p13.txt', 'r') as f:
            addition = 0
            for l in f.readlines():
                addition += int(l)

            print(str(addition)[:10])
    elif arg == '14':
        pass
    else:
        raise ValueError('Not Implemented')


def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    except getopt.error:
        print(getopt.error.msg)
        print('for help use --help')
        sys.exit(2)
    # process options
    for o in opts:
        if o in ('-h', '--help'):
            print(__doc__)
            sys.exit(0)

    # process arguments
    for arg in args:
        process(arg)


if __name__ == '__main__':
    sys.exit(main())
