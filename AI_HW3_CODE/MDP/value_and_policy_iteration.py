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
    U_func_prev = U_func = U_init.astype(float)
    # TODO: check that in case gamma = 1 we don't want to finish runnning here
    while True:
        delta = 0
        U_func_prev = np.copy(U_func)
        for s, _ in np.ndenumerate(U_init):
            if mdp.board[s[0]][s[1]] == 'WALL':
                continue
            if s in mdp.terminal_states:
                U_func[s[0]][s[1]] = float(mdp.board[s[0]][s[1]])
                continue
            next_states = [mdp.step(s, a) for a in mdp.actions.keys()]
            U_s_tag = [U_func_prev[s_tag[0]][s_tag[1]] for s_tag in next_states]
            U_func[s[0]][s[1]] = float(mdp.board[s[0]][s[1]]) + mdp.gamma * max(U_s_tag) 
            if abs(U_func[s[0]][s[1]] - U_func_prev[s[0]][s[1]]) > delta: 
                delta = abs(U_func[s[0]][s[1]] - U_func_prev[s[0]][s[1]])
        if delta < (epsilon * (1 - mdp.gamma)) / mdp.gamma or (delta == 0 and mdp.gamma == 1):
            break             
    return U_func


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy_board = U.astype(str)
    for s, _ in np.ndenumerate(U):
        if s in mdp.terminal_states:
            continue
        next_states = [mdp.step(s, a) for a in mdp.actions.keys()]
        U_s_tag = [U[s_tag[0]][s_tag[1]] for s_tag in next_states]
        keys_list = list(mdp.actions)
        policy_board[s[0], s[1]] = keys_list[np.argmax(U_s_tag)]
    return policy_board


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    utility_board = np.zeros(policy.shape).astype(float)
    for _ in range(policy.shape[0]*policy.shape[1]):
        for s, _ in np.ndenumerate(policy):
            if s in mdp.terminal_states:
                utility_board[s[0]][s[1]] = float(mdp.board[s[0]][s[1]])
                continue
            if mdp.board[s[0]][s[1]] == 'WALL':
                utility_board[s[0]][s[1]] = 0.0
                continue
            s_tag =  mdp.step(s, policy[s[0]][s[1]])
            U_s_tag = utility_board[s_tag[0]][s_tag[1]]
            utility_board[s[0]][s[1]] = float(mdp.board[s[0]][s[1]]) + mdp.gamma * U_s_tag
    return utility_board
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = policy_init
    changed = True
    while changed:
        changed = False
        U = policy_evaluation(mdp, np.array(optimal_policy))
        # mdp.print_utility(U)
        for s, _ in np.ndenumerate(U):
            if s in mdp.terminal_states or mdp.board[s[0]][s[1]] == 'WALL':
                continue
            # check if other actions from s are better than the current policy
            next_states = [mdp.step(s, a) for a in mdp.actions.keys()]
            U_s_tags = [U[s_tag[0]][s_tag[1]] for s_tag in next_states]
            
            # calculate for current optimal_policy
            curr_policy_action = optimal_policy[s[0]][s[1]]
            s_tag_policy = mdp.step(s, curr_policy_action)
            if np.max(U_s_tags) > U[s_tag_policy[0]][s_tag_policy[1]]:
                keys_list = list(mdp.actions)
                optimal_policy[s[0]][s[1]] = keys_list[np.argmax(U_s_tags)]
                changed = True
        # mdp.print_policy(optimal_policy)
    return optimal_policy

