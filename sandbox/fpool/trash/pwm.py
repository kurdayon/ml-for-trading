from mono_dyn import w_mean, w_opos, correct, full_correct

def solution_to_ntps(solution, level_func):
    n = len(solution)
    if n <= 2:
        return 0
    ntps = 0
    for i in range(1, n - 1):
        l1 = solution[i - 1]
        l2 = solution[i]
        l3 = solution[i + 1]

        if l2 > l1 and l2 > l3:
            ntps += 1

        if l2 < l1 and l2 < l3:
            ntps += 1
    return ntps


def mono_dyn(ys, level_func, max_ntps):
    '''
    max_ntps - maximal number of turning points.
    '''
    solutions = [[[ys[0]]]]
    for i in range(1, len(ys)):

        new_solutions = []
        for solution in solutions:

            new_solution = solution[:] + [[ys[i]]]
            new_solutions.append(new_solution[:])

            new_solution = full_correct(new_solution, level_func)
            if new_solution not in new_solutions:
                new_solutions.append(new_solution[:])

        # remove solutions that have to many turning poins
        solutions = [s for s in new_solutions if solution_to_ntps(s, level_func) <= max_tmps]

        # remove solutions that are not pareto optimal


    # now we are done with adding points, we need to take the best solution
