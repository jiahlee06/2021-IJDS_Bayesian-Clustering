from breakpoints import find_break_points
import numpy as np
import pandas as pd
from gurobipy import Model as gurobi_Model, GRB, quicksum
from mosek.fusion import Model as mosek_Model, Variable, Expr, Domain, ObjectiveSense
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans
from itertools import product
from IPython import embed


class MixIntegerGaussianMixture:
    def __init__(self, data, num_component, discrepancy="KS", random_seed=None, p=10, m=100):
        self.data = np.array(data)
        assert len(data.shape) == 2, "Expected 2D array. Reshape your data either using array.reshape(-1, 1) " \
                                     "if your data has a single feature or " \
                                     "array.reshape(1, -1) if it contains a single sample."

        self.num_samples, self.num_features = self.data.shape
        self.num_components = num_component
        self.num_directions = self.num_features ** 2

        # draw projected directions randomly
        rng = np.random.RandomState(random_seed)
        self.projected_directions = self.sample_unit_hyperspheres(self.num_directions, self.num_features, rng)

        # self.projected_data shape: (#directions, #samples)
        self.projected_data = np.dot(self.projected_directions, self.data.transpose())

        # need to iterate each direction to get the projected means and projected covariances.
        self.projected_means = np.zeros((self.num_directions, self.num_components))
        self.projected_weights = np.zeros((self.num_directions, self.num_components))
        self.projected_cov = np.zeros((self.num_directions, self.num_components))
        self.permutation = np.zeros((self.num_directions, self.num_features, self.num_features))

        # after getting the projected mean, covariance and permutation, begin restore multivariate means and covariance
        self.means = np.zeros((self.num_components, self.num_features))
        self.covariances = np.zeros((self.num_components, self.num_features, self.num_features))
        self.weights = np.zeros(self.num_components)

        # generating the breaking points
        self.p, self.m = p, m
        self.Phi, self.v, self.pwl_error = find_break_points(p=self.p, m=self.m)
        # add two large points in case of extrapolation
        self.p = self.p + 2
        assert len(self.Phi) == self.p

        assert discrepancy == "KS" or discrepancy == "TV", "please enter correct discrepancy"
        self.discrepancy = discrepancy

    # generate unit norm random projection directions
    def sample_unit_hyperspheres(self, num_points, num_dim, random_state=None):
        # proof of equally distribution in http://mathworld.wolfram.com/SpherePointPicking.html
        if random_state is not None:
            vec = random_state.randn(num_points, num_dim)
        else:
            vec = np.random.randn(num_points, num_dim)
        vec /= np.linalg.norm(vec, axis=1).reshape(num_points, -1)
        return vec

    def optimize(self, rel_tol=1e-3, timelimit=60, verbose=False):
        # abbreviate proj stand for specific value corresponding to each direction.
        for dd in range(self.num_directions):
            print("current direction: {:2d} / {:2d}".format(dd+1, self.num_directions))
            # the data is: projected_data[dd, :]
            hist_likelihood = []

            # initialize the proj_means, proj_stds and proj_weights
            proj_data = np.sort(self.projected_data[dd, :]).reshape(-1, 1)
            predicted_label = KMeans(n_clusters=self.num_components, random_state=0).fit_predict(proj_data)
            df = np.concatenate([predicted_label.reshape(-1, 1), proj_data], axis=1)
            df = pd.DataFrame(df, columns=["label", "proj_data"])
            group_df = df.groupby("label")["proj_data"]
            proj_means = np.array(group_df.mean())
            proj_stds = np.array(group_df.std())
            proj_weights = np.array(group_df.count()/self.num_samples)

            # proj_means = np.array([0.08903399, 9.92190726, 4.92198972])
            # proj_stds = np.array([1.05461802, 0.85012444, 0.95667114])
            # proj_weights = np.array([0.33658847, 0.32383306, 0.33957847])

            while True:
                proj_means, proj_stds = self.param_model(
                    proj_data=proj_data, init_proj_means=proj_means, init_proj_stds=proj_stds,
                    proj_weights=proj_weights, verbose=verbose, timelimit=timelimit)

                proj_weights = self.weight_model(
                    proj_data=proj_data, proj_means=proj_means, proj_stds=proj_stds, verbose=verbose)

                likelihood = self.compute_likelihood(proj_data, proj_means, proj_stds, proj_weights)
                hist_likelihood.append(likelihood)
                if len(hist_likelihood) >= 2:
                    # if hist_likelihood[-1] < hist_likelihood[-2]:
                    #     print("likelihood decrease, need more time to run")
                    error = abs(hist_likelihood[-1] - hist_likelihood[-2]) / abs(hist_likelihood[-2])
                    if error < rel_tol:
                        break
            self.projected_means[dd, :] = proj_means
            self.projected_cov[dd, :] = proj_stds
            self.projected_weights[dd, :] = proj_weights
            if verbose:
                print(hist_likelihood)
                print("=" * 100)

        self.weights = np.mean(self.projected_weights, axis=0)

        self.means, self.permutations = self.multivariate_mean(self.projected_means, self.projected_directions)
        self.covariances = self.multivariate_covariance(self.projected_cov, self.projected_directions, self.permutations)
        return self.means, self.covariances, self.weights


    def build_param_model(self, proj_data):
        param_model = gurobi_Model("model for solving parameter")
        vec_t = param_model.addVars(self.num_components, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="mu_over_std")
        vec_s = param_model.addVars(self.num_components, vtype=GRB.CONTINUOUS, lb=0.0, name="inverse_of_std")
        epsilon = param_model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="epsilon")
        Z = param_model.addVars(self.num_components, self.num_samples, self.p - 1, vtype=GRB.BINARY,
                                     name="Z")
        Y = param_model.addVars(self.num_components, self.num_samples, self.p, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0,
                                     name="Y")

        for i, j in product(range(self.num_components), range(self.num_samples)):
            param_model.addConstr(
                vec_s[i] * proj_data[j] - vec_t[i] == quicksum([self.v[k] * Y[i, j, k] for k in range(self.p)])
                , name="linear_approximation"
            )

            param_model.addConstr(Y[i, j, 0] <= Z[i, j, 0], name="specify_the_invterval_prob_1")

            param_model.addConstrs(
                (Y[i, j, k] <= Z[i, j, k - 1] + Z[i, j, k] for k in range(1, self.p - 1))
                , name="specify_the_invterval_prob_2"
            )

            param_model.addConstr(Y[i, j, self.p - 1] <= Z[i, j, self.p - 2], name="specify_the_invterval_prob_3")
            param_model.addConstr(quicksum([Z[i, j, k] for k in range(self.p - 1)]) == 1, name="interval_constraint")
            param_model.addConstr(quicksum([Y[i, j, k] for k in range(self.p)]) == 1, name="probability_constraint")

        param_model.setObjective(epsilon, GRB.MINIMIZE)

        # store these variables
        param_model._vec_t = vec_t
        param_model._vec_s = vec_s
        param_model._epsilon = epsilon
        param_model._Y = Y
        param_model._Z = Z

        return param_model

    def param_model(self, proj_data, init_proj_means, init_proj_stds, proj_weights, timelimit=60, verbose=False):
        # shape of proj_data: (#samples, 1)
        assert np.all(np.diff(proj_data, axis=0) >= 0), "data must be sorted"
        assert len(proj_weights) == self.num_components
        if not abs(sum(proj_weights) - 1) < 1e-3 or not np.all(proj_weights > 0):
            print("weight of component maybe wrong:")
            print(proj_weights)

        # print("update the weight:", proj_weights)

        param_model = self.build_param_model(proj_data)

        # -------------- initial set up
        if init_proj_stds is not None and init_proj_means is not None:
            # the primary computational variable is the binary variable, so we only initialize Z
            # and actually, most of time, it can not yield a new incumbent solution for the lazy callback
            z_score = (proj_data.reshape(1, self.num_samples) - init_proj_means.reshape(self.num_components, 1))\
                / init_proj_stds.reshape(self.num_components, 1)
            z_score = z_score.reshape(self.num_components, self.num_samples, 1)

            component_index, data_index, break_point_index = \
                np.where(np.logical_and(self.v[0:-1] < z_score, z_score <= self.v[1:]))

            for i, j, k in product(range(self.num_components), range(self.num_samples), range(self.p - 1)):
                param_model._Z[i, j, k].start = 0
            for i, j, k in zip(component_index, data_index, break_point_index):
                param_model._Z[i, j, k].start = 1

            # first solve to get Y and want to add at least one epsilon constraint. otherwise, best obj will equal to 0
            # param_model.update()
            # prepared_model = param_model.copy()
            # for i, j, k in product(range(self.num_components), range(self.num_samples), range(self.p - 1)):
            #     prepared_model.getVarByName("Z[{},{},{}]".format(i, j, k)).start = 0
            # for i, j, k in zip(component_index, data_index, break_point_index):
            #     prepared_model.getVarByName("Z[{},{},{}]".format(i, j, k)).start = 1
            # prepared_model.params.OutputFlag = 0
            # prepared_model.optimize()
            #
            # tempY = [prepared_model.getVarByName("Y[{},{},{}]".format(i, j, k)).X
            #          for i, j, k in product(range(self.num_components), range(self.num_samples), range(self.p))]
            # tempY = np.array(tempY).reshape((self.num_components, self.num_samples, self.p))
            #
            # tempZ = [prepared_model.getVarByName("Z[{},{},{}]".format(i, j, k)).X
            #          for i, j, k in product(range(self.num_components), range(self.num_samples), range(self.p - 1))]
            #
            # tempZ = np.array(tempZ).reshape((self.num_components, self.num_samples, self.p - 1))
            #
            # c_index, d_index, b_index = np.where(tempZ != 0)
            # assert np.all(c_index == component_index) and np.all(d_index == data_index) \
            #     and np.all(b_index == break_point_index)
        # -------------- initial set up finish

        if self.discrepancy == "KS":
            inactive_set = list(np.arange(self.num_samples))
            # # add the constraint which has the largest discrepancy.
            # inactive_set = np.array(inactive_set)
            # KS_dist = (inactive_set + 1) / self.num_samples - np.sum(
            #     np.sum(self.Phi * tempY[:, inactive_set, :], axis=2)
            #     * proj_weights.reshape(self.num_components, 1), axis=0)
            #
            # max_ind = np.argsort(abs(KS_dist))[-1]
            # constraint_ind = inactive_set[max_ind]
            # param_model.addConstrs(
            #     ((j + 1) / self.num_samples - quicksum(
            #         [proj_weights[i] * self.Phi[k] * param_model._Y[i, j, k]
            #          for i in range(self.num_components) for k in range(self.p)]) <= param_model._epsilon
            #      for j in [constraint_ind]), name="bound_objective_function_1"
            # )
            #
            # param_model.addConstrs(
            #     ((j + 1) / self.num_samples - quicksum(
            #         [proj_weights[i] * self.Phi[k] * param_model._Y[i, j, k]
            #          for i in range(self.num_components) for k in range(self.p)]) >= -param_model._epsilon
            #      for j in [constraint_ind]), name="bound_objective_function_1"
            # )
            # inactive_set = list(inactive_set)
            # inactive_set.remove(constraint_ind)
            # print("add constraint:", constraint_ind)
            # print("-"*50)
            # print("-"*50)

        else:
            inactive_set = list(map(lambda x: list(x), np.triu_indices(n=self.num_samples, k=1)))


        # store three important variables for callback function
        param_model._proj_weights = proj_weights
        param_model._inactive_set = inactive_set
        param_model._pwl_error = self.pwl_error

        param_model._num_samples = self.num_samples
        param_model._p = self.p
        param_model._num_components = self.num_components
        param_model._Phi = self.Phi
        param_model._v = self.v


        if not verbose:
            param_model.params.OutputFlag = 0
        param_model.params.TimeLimit = timelimit
        param_model.params.MIPGap = 1e-5
        param_model.params.Heuristics = 0.05 if timelimit < np.inf else 1

        param_model.params.MIPGapAbs = min(self.pwl_error, 1)
        param_model.Params.LazyConstraints = 1
        # param_model.Params.PreSolve = 0
        # param_model.Params.PreCrush = 1

        if self.discrepancy == "KS":
            param_model.optimize(MixIntegerGaussianMixture.check_KS_constraint)
        else:
            param_model.optimize(MixIntegerGaussianMixture.check_TV_constraint)

        if param_model.status == GRB.INTERRUPTED:
            vec_t = param_model._break_vec_t.values()
            vec_s = param_model._break_vec_s.values()
        else:
            vec_t = [t.X for t in param_model._vec_t.values()]
            vec_s = [s.X for s in param_model._vec_s.values()]
        assert np.all(np.array(vec_s) > 0), "need more time to run"
        proj_means = np.array(vec_t) / np.array(vec_s)
        proj_stds = 1 / np.array(vec_s)
        return proj_means, proj_stds

    @staticmethod
    def check_KS_constraint(model, where):
        if where == GRB.Callback.MIPSOL and len(model._inactive_set) > 0:
            inactive_set = np.array(model._inactive_set)

            proj_weights = np.array(model._proj_weights)
            Y = model.cbGetSolution(model._Y).values()
            Y = np.array(list(Y)).reshape(model._num_components, model._num_samples, model._p)
            KS_dist = (np.arange(model._num_samples) + 1) / model._num_samples - np.sum(
                np.sum(model._Phi * Y, axis=2) * proj_weights.reshape(model._num_components, 1), axis=0)

            max_inactive_dist = max(abs(KS_dist[inactive_set]))
            max_inactive_ind = np.argsort(abs(KS_dist[inactive_set]))[-1]

            if model.cbGet(GRB.callback.MIPSOL_OBJBST) < max_inactive_dist:
                constraint_ind = model._inactive_set[max_inactive_ind]

                if model.params.OutputFlag:
                    print("max dist: {:.4f} max index: {:.0f}".format(max_inactive_dist, max_inactive_ind))
                    print("obj best: {:.4f} obj bound: {:.0f}"
                          .format(model.cbGet(GRB.callback.MIPSOL_OBJBST), model.cbGet(GRB.callback.MIPSOL_OBJBND)))
                    print("Add Lazy constraints: {:d},  max active dist: {:.4f}, max dist: {:.4f}, #inactive: {:d}"
                          .format(constraint_ind, max_inactive_dist, max(abs(KS_dist)), len(model._inactive_set)))

                model._inactive_set.remove(constraint_ind)

                model.cbLazy(
                    (constraint_ind + 1)/model._num_samples - quicksum(
                        [proj_weights[i] * model._Phi[k] * model._Y[i, constraint_ind, k]
                         for i in range(model._num_components) for k in range(model._p)]) <= model._epsilon
                )

                model.cbLazy(
                    (constraint_ind+1)/model._num_samples - quicksum(
                        [proj_weights[i] * model._Phi[k] * model._Y[i, constraint_ind, k]
                         for i in range(model._num_components) for k in range(model._p)]) >= -model._epsilon
                )

            if max(abs(KS_dist)) < model._pwl_error:
                # there have no need to run anymore, for the error of approximation have already reach that.
                # can not use model.terminate(), for it will return GRB_INTERRUPTED.
                model._break_vec_t = model.cbGetSolution(model._vec_t)
                model._break_vec_s = model.cbGetSolution(model._vec_s)
                if model.params.OutputFlag:
                    print("=="*50)
                    print("all of the distances are smaller than piecewise linear error. Terminate!")
                    print("==" * 50)
                model.terminate()
                # model.Params.TimeLimit = model.getAttr(GRB.Attr.Runtime)


    @staticmethod
    def check_TV_constraint(model, where):
        if where == GRB.Callback.MIPSOL and len(model._inactive_set) > 0:
            inactive_set = np.array(model._inactive_set)

            proj_weights = np.array(model._proj_weights)
            Y = model.cbGetSolution(model._Y).values()
            Y = np.array(list(Y)).reshape(model._num_components, model._num_samples, model._p)

            inactive_dist = (inactive_set[0] - inactive_set[1]) / model._num_samples - np.sum(
                np.sum(model._Phi * (Y[:, model._inactive_set[0], :] - Y[:, model._inactive_set[1], :]), axis=2)
                * proj_weights.reshape(model._num_components, 1), axis=0)

            all_set = np.triu_indices(n=model._num_samples, k=1)
            TV_dist = (all_set[0] - all_set[1]) / model._num_samples - np.sum(
                np.sum(model._Phi * (Y[:, all_set[0], :] - Y[:, all_set[1], :]), axis=2)
                * proj_weights.reshape(model._num_components, 1), axis=0)

            max_inactive_ind = np.argmax(np.abs(inactive_dist))
            max_inactive_dist = abs(inactive_dist[max_inactive_ind])
            if model.cbGet(GRB.callback.MIPSOL_OBJBST) < max_inactive_dist:
                ind_i, ind_j = model._inactive_set[0][max_inactive_ind], model._inactive_set[1][max_inactive_ind]
                if model.params.OutputFlag:
                    print("max dist: {:.4f} max index: {:.0f}".format(max_inactive_dist, max_inactive_ind))
                    print("obj best: {:.4f} obj bound: {:.0f}"
                          .format(model.cbGet(GRB.callback.MIPSOL_OBJBST), model.cbGet(GRB.callback.MIPSOL_OBJBND)))
                    print("Add Lazy constraints: ({:d},{:d}),  max active dist: {:.4f}, max dist: {:.4f}, #inactive: {:d}"
                          .format(ind_i + 1, ind_j + 1, max_inactive_dist, max(abs(TV_dist)), len(model._inactive_set[0])))

                model.cbLazy(
                    (ind_j - ind_i)/model._num_samples - quicksum(
                        [proj_weights[l] * model._Phi[k] * (model._Y[l, ind_j, k] - model._Y[l, ind_i, k])
                         for l in range(model._num_components) for k in range(model._p)]) <= model._epsilon
                )

                model.cbLazy(
                    (ind_j - ind_i)/model._num_samples - quicksum(
                        [proj_weights[l] * model._Phi[k] * (model._Y[l, ind_j, k] - model._Y[l, ind_i, k])
                         for l in range(model._num_components) for k in range(model._p)]) >= -model._epsilon
                )

                del model._inactive_set[0][max_inactive_ind]
                del model._inactive_set[1][max_inactive_ind]



            if max(abs(TV_dist)) < model._pwl_error:
                model._break_vec_t = model.cbGetSolution(model._vec_t)
                model._break_vec_s = model.cbGetSolution(model._vec_s)
                if model.params.OutputFlag:
                    print("==" * 50)
                    print("all of the distances are smaller than piecewise linear error. Terminate!")
                    print("==" * 50)
                model.terminate()

    def weight_model(self, proj_data, proj_means, proj_stds, verbose=False):
        weight_model = gurobi_Model("model for solving weights")
        epsilon = weight_model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="objective_function")
        proj_weights = weight_model.addVars(self.num_components, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0
                                            , name="component_weight")

        coeff = norm.cdf((proj_data.reshape(-1, 1) - proj_means) / proj_stds)

        weight_model.addConstrs(
            (
                (i+1) / self.num_samples -
                quicksum([proj_weights[j] * coeff[i, j] for j in range(self.num_components)]) <= epsilon
                for i in range(self.num_samples)
             ), name="objective_constraint_1")

        weight_model.addConstrs(
            (
                (i+1) / self.num_samples -
                quicksum([proj_weights[j] * coeff[i, j] for j in range(self.num_components)]) >= -epsilon
                for i in range(self.num_samples)
            ), name="objective_constraint_2")

        weight_model.addConstr(proj_weights.sum() == 1, name="component_sum_constraint")

        weight_model.setObjective(epsilon, GRB.MINIMIZE)

        if not verbose:
            weight_model.params.OutputFlag = 0
        weight_model.params.OutputFlag = 0
        weight_model.optimize()

        proj_weights = [t.X for t in proj_weights.values()]
        return np.array(proj_weights)

    def compute_likelihood(self, proj_data, proj_means, proj_stds, proj_weights):
        exp_part = (proj_data.reshape(-1, 1) - proj_means) ** 2 / (2 * proj_stds ** 2)
        exp_part = np.exp(-exp_part)
        coeff_part = proj_weights / np.sqrt(2 * np.pi * proj_stds ** 2)
        likelihood = coeff_part * exp_part
        likelihood = np.sum(np.log(np.sum(likelihood, axis=1)))
        return likelihood

    # using the result got from the above to restore multivariate means
    def multivariate_mean(self, projected_means, projected_directions):
        """
        :param projected_means: shape: (#directions, #components)
        :param projected_directions: shape: (#directions, #features)
        in paper, d = #features; d^2 = #directions
        :return:
        """
        assert projected_means.shape[0] == projected_directions.shape[0]
        num_components = projected_means.shape[1]
        num_directions = projected_directions.shape[0]
        num_features = projected_directions.shape[1]
        M = mosek_Model("solve multivariate mean")
        obj = M.variable(num_directions, Domain.greaterThan(0))
        mat_P = M.variable([num_directions, num_components, num_components], Domain.binary())
        mat_mean = M.variable([num_components, num_features], Domain.unbounded())
        one = [1] * num_components
        for dd in range(num_directions):
            single_P = mat_P.slice([dd, 0, 0], [dd + 1, num_components, num_components])
            single_P = single_P.reshape(num_components, num_components)
            M.constraint(Expr.mul(single_P, one), Domain.equalsTo(1))
            M.constraint(Expr.mul(single_P.transpose(), one), Domain.equalsTo(1))
            temp = Expr.sub(Expr.mul(single_P, projected_means[dd, :]), Expr.mul(mat_mean, projected_directions[dd, :]))
            M.constraint(Expr.vstack(0.5, obj.index(dd), temp), Domain.inRotatedQCone())
        M.objective(ObjectiveSense.Minimize, Expr.sum(obj))
        M.solve()
        mip_means = mat_mean.level().reshape(num_components, num_features)
        permutations = mat_P.level().reshape(num_directions, num_components, num_components)
        return mip_means, permutations

    # using the result got from the above to restore multivariate covariances
    def multivariate_covariance(self, projected_cov, projected_directions, permutations):
        """
        :param projected_cov: shape: (#directions, #components)
        :param projected_directions: shape: (#directions, #features)
        :param permutations: shape: (#directions, #num_components, #num_components)
        :return:
        """
        model = mosek_Model("solve multivariate covariance matrix")
        assert projected_cov.shape[0] == projected_directions.shape[0]
        num_components = projected_cov.shape[1]
        num_directions = projected_directions.shape[0]
        num_features = projected_directions.shape[1]

        mat_cov = model.variable(Domain.inPSDCone(num_features, num_components))
        obj = model.variable([num_directions, num_components], Domain.greaterThan(0))

        for dd in range(num_directions):
            for k in range(num_components):
                Ps = permutations[dd, k, :] @ projected_cov[dd, :]
                rho = projected_directions[dd, :].reshape((num_features, 1))
                Sigma = mat_cov.slice([k, 0, 0], [k + 1, num_features, num_features]).reshape(num_features,
                                                                                              num_features)
                quad = Expr.mul(Expr.mul(rho.transpose(), Sigma), rho)

                model.constraint(Expr.add(obj.index(dd, k), quad), Domain.greaterThan(Ps))
                model.constraint(Expr.sub(obj.index(dd, k), quad), Domain.greaterThan(-Ps))

        model.objective(ObjectiveSense.Minimize, Expr.sum(obj))
        model.solve()
        mip_covariances = mat_cov.level().reshape((num_components, num_features, num_features))
        return mip_covariances


if __name__ == "__main__":
    random_seed = 43

    # edit here
    in_csv = "processed/brca.csv"
    out_csv = "model/bandi_brca.csv"
    col_names = ['worst_area', 'worst_smoothness', 'mean_texture']
    num_components = 2
    d = 3
    
    # import data / edit here
    X_df = pd.read_csv(in_csv, header = 0)
    features = np.array(X_df[col_names])
    labels = np.array(X_df[['diagnosis']])
    
    data = np.concatenate([features, labels.reshape(-1, 1)], axis=1)
    columns = ["x{}".format(i + 1) for i in range(features.shape[1])]
    data = pd.DataFrame(data, columns=[*columns, "y"])

    # estimate parameters

    model = MixIntegerGaussianMixture(data=features, num_component=num_components, random_seed=random_seed)
    mip_means, mip_covariances, mip_weights = model.optimize(verbose=False, timelimit=1800)

    print("mip means:", mip_means)
    print("mip covs are:", mip_covariances)
    print("mip weights are:", mip_weights)    
    
    # assignment based on parameters values    
    bayes_p = [np.log(mip_weights[k]) + np.log(np.linalg.det(mip_covariances[k])**(-0.5)) - 0.5*((features[i] - mip_means[k]) @ np.linalg.pinv(mip_covariances[k]) @ (features[i] - mip_means[k]).T) for i, k in product(range(features.shape[0]), range(num_components))]
    bayes_p = np.array(bayes_p).reshape(features.shape[0],num_components)
    z = np.zeros(features.shape[0])
    for i in range(features.shape[0]):
        max_value = max(bayes_p[i])
        z[i] = list(bayes_p[i]).index(max_value)

    # export outcomes
    mod_df = pd.concat([pd.DataFrame({'param': ["z_%03d" % i for i in range(1, len(z)+1)],
                                'value': z.astype('int')}),
                     pd.DataFrame({'param': ["m_%01d-%01d" % (k, d) for (k, d) in product(range(1, num_components+1), range(1, d+1))],
                                'value': mip_means.reshape(-1)}),
                     pd.DataFrame({'param': ["cov_%01d-%01d-%01d" % (k, d1, d2) for (k, d1, d2) in product(range(1, num_components+1), range(1, d+1), range(1, d+1))],
                                'value': mip_covariances.reshape(-1)}),
                     pd.DataFrame({'param': ["p_%01d" % i for i in range(1, len(mip_weights)+1)],
                                'value': mip_weights})],
                    axis=0, ignore_index=True)
    mod_df.to_csv(out_csv, header=True, index=False)
    
    

    
    
