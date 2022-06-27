from copy import deepcopy
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    U_init = np.array(U_init)
    U_func = deepcopy(U_init.astype(float))
    # TODO: check that in case gamma = 1 we don't want to finish runnning here
    while True:
        delta = 0
        U_func_prev = deepcopy(U_func)
        for s, _ in np.ndenumerate(U_init):
            if mdp.board[s[0]][s[1]] == 'WALL':
                continue
            if s in mdp.terminal_states:
                U_func[s[0]][s[1]] = float(mdp.board[s[0]][s[1]])
                continue
            
            U_func[s[0]][s[1]] = calcU_tag(mdp, U_func_prev, s, mdp.actions)

            if abs(U_func[s[0]][s[1]] - U_func_prev[s[0]][s[1]]) > delta: 
                delta = abs(U_func[s[0]][s[1]] - U_func_prev[s[0]][s[1]])
        if delta < (epsilon * (1 - mdp.gamma)) / mdp.gamma or (delta == 0 and mdp.gamma == 1):
            break             
    return U_func


def calcU_s_tag(mdp, U, state, action):
    transitions = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}
    all_prob_U = []
    for a in mdp.actions:
        new_s = mdp.step(state, a)
        U_new_s = U[new_s[0]][new_s[1]]
        all_prob_U.append(mdp.transition_function[action][transitions[a]] * U_new_s)
    return sum(all_prob_U) 

def calcU_tag(mdp, U, state, actions):
    all_U_s_tags = []
    for a in actions:
        all_U_s_tags.append(calcU_s_tag(mdp, U, state, a))
    return float(mdp.board[state[0]][state[1]]) + mdp.gamma * max(all_U_s_tags) 


def best_U_in_town(mdp, U, state):
    max_a = None
    max_U = -float("inf")
    for a in mdp.actions:
        u = calcU_tag(mdp, U, state, [a])
        if u > max_U:
            max_a = a
            max_U = u
    return max_U, max_a
    

def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy_board = U.astype(str)
    for s, _ in np.ndenumerate(U):
        if s in mdp.terminal_states or mdp.board[s[0]][s[1]] == 'WALL':
            continue
        _, max_a = best_U_in_town(mdp, U, s)
        
        policy_board[s[0]][s[1]] = max_a

    return policy_board


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    utility_board = [[0, 0, 0, 0] for i in range(3)]
    while True:
        delta  = 0 
        U_next = deepcopy(utility_board)
        for s, _ in np.ndenumerate(policy):
            if mdp.board[s[0]][s[1]] == 'WALL':
                U_next[s[0]][s[1]] = 0.0
                continue
            
            elif s in mdp.terminal_states:
                U_next[s[0]][s[1]] = float(mdp.board[s[0]][s[1]])
                
            else:
                U_next[s[0]][s[1]] = calcU_tag(mdp, utility_board, s, mdp.actions)
            if delta < abs(utility_board[s[0]][s[1]] - U_next[s[0]][s[1]]):
                delta = abs(utility_board[s[0]][s[1]] - U_next[s[0]][s[1]])
        utility_board = deepcopy(U_next)
        if delta <= 0 :
            break
    return utility_board


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    # v
    optimal_policy = deepcopy(policy_init)
    changed = True
    while changed:
        changed = False
        U = policy_evaluation(mdp, optimal_policy)
        # mdp.print_utility(U)
        for s, _ in np.ndenumerate(U):
            if s in mdp.terminal_states or mdp.board[s[0]][s[1]] == 'WALL':
                continue
            # check if other actions from s are better than the current policy
            max_U, max_a = best_U_in_town(mdp, U, s)
            # calculate for current optimal_policy
            if max_U > calcU_tag(mdp, U, s, [optimal_policy[s[0]][s[1]]]):
                optimal_policy[s[0]][s[1]] = max_a
                changed = True
        # mdp.print_policy(optimal_policy)
    return optimal_policy
