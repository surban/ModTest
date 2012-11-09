import svm, svmutil
import matplotlib.pyplot as plt

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
    #base = [1, 3, 5]
    base = [1, 3, 5, 7, 13, 27]
    divisors = base + [9, 12, 15, 2, 20]
    #divisors = [1, 2, 3, 5]
    limits = [sum(base)] + base
    factors_set = generate_factors_set(len(base))
    sums = [sum_factors(base, factors) for factors in factors_set]
    modulos_set = [calculate_modulos(divisors, s) for s in sums]
    input_set = [[s] + modulos for modulos, s in zip(modulos_set, sums)]
    target_set = [[factors[i] for factors in factors_set] 
                  for i in range(len(base))]

    print "base:         ", base
    print "factors_set:  ", factors_set
    print "sums:         ", sums
    print "modulos_set:  ", modulos_set
    print "input_set:    ", input_set
    print "target_set:   ", target_set

    n_success = 0
    n_failure = 0
    total_correct = 0
    for n in range(len(target_set)):
        targets = [to_svm_label(t) for t in target_set[n]]

        print
        print "Class (base element): %d = %d" % (n, base[n])
        print "targets:       ", targets
        print "input vectors: ", input_set
        
        problem = svmutil.svm_problem(targets, input_set)
        params = svmutil.svm_parameter("-q -s 0 -t 0 -c 100")
        classifier = svmutil.svm_train(problem, params)
        labels, acc, _ = svmutil.svm_predict(targets, input_set, classifier)
        correct = acc[0]
        total_correct += correct
        if correct == 100:
            print "Success"
            n_success += 1
        else:
            print "Failure"
            n_failure += 1
            #class_plot(input_set, targets, [2, 3], limits)
            #plt.show()

    print
    print "Successes: %d, Failures: %d" % (n_success, n_failure)
    print "Total accuracy: %f %%" % (total_correct / len(target_set))

        

    

