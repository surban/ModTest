import svm, svmutil
import matplotlib.pyplot as plt
import sys

from itertools import product

def test_base(base):
    factor_set = list(product([0,1], repeat=len(base)))
    print "factor_set: ", factor_set
    sums = [sum_factors(base, factors) for factors in factor_set]
    print "sums: ", sums
    base_occurences = [0 for _ in base]
    for s in sums:
        if s in base:
            base_occurences[base.index(s)] += 1
            if base_occurences[base.index(s)] > 1:
                print "base entry %d is constructable" % s
                return False
    return True

def sv_to_vec(sv, length):
    vec = numpy.zeros(length)
    for indx, val in sv.iteritems():
        if indx != -1:
            vec[indx-1] = val
    return vec
    
def generate_factors_set(n):
    if n == 1:
        return [[0], [1]]
    else:
        fs = generate_factors_set(n-1)
        fs0 = [[0] + f for f in fs]
        fs1 = [[1] + f for f in fs]
        return fs0 + fs1

def sum_factors(base, factors):
    s = 0
    for b, f in zip(base, factors):
        s += b * f
    return s

def calculate_modulos(modulos, x):
    return [x % m for m in modulos]

def calculate_modulos_with_shifts(modulos, shifts, x):
    return [(x+s) % m for m, s in zip(modulos, shifts)]

def take_components(x, components):
    if components is not None:
        return [x[c] for c in components]
    else:
        return x

def vectors_to_xy(vs):
    xs = [v[0] for v in vs]
    ys = [v[1] for v in vs]
    return xs, ys

def class_plot(xs, targets, components=None, limits=None):           
    xs = [take_components(x, components) for x in xs]
    limits = take_components(limits, components)

    xpos = [x for x, c in zip(xs, targets) if c > 0]
    xneg = [x for x, c in zip(xs, targets) if c < 0]

    print "xpos: ", xpos
    print "xneg: ", xneg

    plt.clf()
    plt.hold(True)
    px, py = vectors_to_xy(xpos)
    plt.plot(px, py, 'rx')
    px, py = vectors_to_xy(xneg)
    plt.plot(px, py, 'bo')

    plt.xticks(range(20))
    plt.yticks(range(20))
    if limits is not None:
        plt.xlim(-0.2, limits[0]+0.2)
        plt.ylim(-0.2, limits[1]+0.2)
    



def to_svm_label(l):
    if l > 0.5:
        return 1
    else:
        return -1

if __name__ == '__main__':
    base = [1, 3, 5]
    base = [1, 3, 5, 7, 14]
    divisors = base
    #divisors = [2, 4, 6, 8, 14, 28]
    #divisors = [1, 2, 3, 5]

    if not test_base(base):
        sys.exit(1)

    limits = [sum(base)] + base
    factors_set = generate_factors_set(len(base))
    sums = [sum_factors(base, factors) for factors in factors_set]
    target_set = [[factors[i] for factors in factors_set] 
                  for i in range(len(base))]
    shift_ranges = [range(d) for d in divisors]
    shifts_set = list(product(*shift_ranges))

    print "base:         ", base
    print "factors_set:  ", factors_set
    print "sums:         ", sums
    print "target_set:   ", target_set

    print "Shift set:"
    #for shift in shifts_set:
    #    print shift

    best_correct_percent = 0 
    for shifts in shifts_set:
        n_success = 0
        n_failure = 0
        total_correct = 0
        for n in range(len(target_set)):
            targets = [to_svm_label(t) for t in target_set[n]]

            modulos_set = [calculate_modulos_with_shifts(divisors, shifts, s) for s in sums]
            input_set = [[s] + modulos for modulos, s in zip(modulos_set, sums)]

            #print
            #print "Class (base element): %d = %d" % (n, base[n])
            #print "targets:       ", targets
            #print "shifts:        ", shifts
            #print "modulos_set:   ", modulos_set
            #print "input_set:     ", input_set
        
            problem = svmutil.svm_problem(targets, input_set)
            params = svmutil.svm_parameter("-q -s 0 -t 0 -c 100")
            classifier = svmutil.svm_train(problem, params)
            labels, acc, _ = svmutil.svm_predict(targets, input_set, classifier)
            correct = acc[0]
            total_correct += correct
            if correct == 100:
                #print "Success"
                n_success += 1
            else:
                #print "Failure"
                n_failure += 1
                #class_plot(input_set, targets, [2, 3], limits)
                #plt.show()

        correct_percent = total_correct / len(target_set)

        #print
        #print "Successes: %d, Failures: %d" % (n_success, n_failure)
        #print "Total accuracy: %f %%" % correct_percent

        if correct_percent > best_correct_percent:
            best_correct_percent = correct_percent
            best_shifts = shifts

        print "Best total accuracy so far: %.3f%% using shift %s" % \
            (best_correct_percent, str(best_shifts))

    print
    print "Best total accuracy: %f %%" % best_correct_percent
    print "Using best shifts:   ", best_shifts


        

    

