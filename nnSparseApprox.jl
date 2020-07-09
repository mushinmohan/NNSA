using LinearAlgebra

function sparseApprox(A, b, k, opts...)
    if length(opts) == 0
        ε = .1
        σ = -2.5
        doprint = false
        print_debug = false
        info_frac = zeros(k)
        for i = 1 : k
            info_frac[i] = (1 / (k - (i - 1)))
        end
    elseif length(opts) == 1
        ε = opts[1]
        σ = -2.5
        doprint = false
        print_debug = false
        info_frac = zeros(k)
        for i = 1 : k
            info_frac[i] = (1 / (k - (i - 1)))
        end
    elseif length(opts) == 2
        ε = opts[1]
        σ = opts[2]
        doprint = false
        print_debug = false
        info_frac = zeros(k)
        for i = 1 : k
            info_frac[i] = (1 / (k - (i - 1)))
        end
    elseif length(opts) == 3
        ε = opts[1]
        σ = opts[2]
        doprint = opts[3]
        print_debug = false
        info_frac = zeros(k)
        for i = 1 : k
            info_frac[i] = (1 / (k - (i - 1)))
        end
    elseif length(opts) == 4
        ε = opts[1]
        σ = opts[2]
        doprint = opts[3]
        print_debug = opts[4]
        info_frac = zeros(k)
        for i = 1 : k
            info_frac[i] = (1 / (k - (i - 1)))
        end
    elseif length(opts) > 4
        ε = opts[1]
        σ = opts[2]
        doprint = opts[3]
        print_debug = opts[4]
        info_frac = opts[5]
    end

    n, numtar = size(A)
    info_bound = info_frac[1] * (1 - ε)
    can_use_prelim = falses(numtar)
    max_expl = zeros(Float64, numtar)
    prohib = Array{Any, 1}(nothing, k)
    prohib_counter = zeros(Int64, k)
    improvement = .03
    counter_prelim = 0;
    bestres = 1
    bestsol = zeros(numtar)
    iter = 1
    δ = [0.1, 0.1]
    tol = 0.001
    lambda = 0.01
    maxiter_gd = 100
    pre_al = 100000
    prohib[1] = zeros(Int64, numtar)
    #combinations of size 1 and 2 are treated as base cases
    tar_norm = norm(b, 1)
    if doprint
        println("Checking combinations of size: 1")
        println("Maximum residual allowed: ", 1 - info_bound)
    end
    for i = 1 : numtar
        tmp = ladConstrained_1d(b, A[:, i], σ)
        tmpsol = zeros(numtar)
        tmpsol[i] = tmp[1]
        tmpres = norm(b - A * tmpsol, 1) / norm(b, 1)
        tmprawres = norm(A * tmpsol, 1)
        max_expl[i] = 1 - tmpres
        if tmpres < bestres
            bestres = tmpres
            bestsol .= tmpsol
        end
        if max_expl[i] >= info_bound
            if doprint
                println(i)
                println("Residual = ", tmpres)
            end
            can_use_prelim[i] = true
            counter_prelim = counter_prelim + 1
            prohib_counter[1] += 1
            prohib[1][prohib_counter[1]] = i
        end
    end
    
    cmblist = zeros(Int64, counter_prelim)
    iter_cmb = 1
    for i = 1 : numtar
        if can_use_prelim[i]
            cmblist[iter_cmb] = i
            iter_cmb += 1
        end
    end
    outer_loop_cap = length(cmblist)
    prohib[2] = zeros(Int64, pre_al, 2)
    can_use_phase2 = Array{Any, 1}(undef, outer_loop_cap)
    for i = 1 : outer_loop_cap
        can_use_phase2[i] = falses(numtar)
    end
    max_expl_phase2 = zeros(Float64, pre_al)


    S = zeros(n, 2)
    info_bound = info_frac[2] * (1 - ε)
    if doprint
        println("# of valid 1-combinations : ", outer_loop_cap)
        println("Checking combinations of size: 2")
        println("Maximum residual allowed: ", 1 - info_bound)
    end
    for i = 1 : outer_loop_cap
        if doprint
            println("Combination# ", i)
        end
        
        for j = 1 : numtar
            if cmblist[i] != j && !can_use_phase2[i][j] && !isprohib_v4([cmblist[i] j], prohib, 2, prohib_counter[2]) &&
                (max_expl[j] + max_expl[cmblist[i]]) >= info_bound

                S[:, 1] .= A[:, cmblist[i]]
                S[:, 2] .= A[:, j]

                tmp = ladFitGd_constrained_v2(S, b, σ, tol, δ, maxiter_gd)
                tmpsol = zeros(numtar)
                tmpsol[cmblist[i]] = tmp[1]
                tmpsol[j] = tmp[2]
                tmpres = norm(b - A * tmpsol, 1) / norm(b, 1)
                
                if tmpres < bestres
                    if doprint
                        println([cmblist[i] j])
                        println("Residual = ", tmpres)
                    end
                    bestres = tmpres
                    bestsol .= tmpsol
                end
                if 1 - (tmpres) >= info_bound
                    can_use_phase2[i][j] = true
                    prohib_counter[2] += 1
                    max_expl_phase2[prohib_counter[2]] = 1 - tmpres
                    prohib[2][prohib_counter[2], 1] = cmblist[i]
                    prohib[2][prohib_counter[2], 2] = j
                end
            end
        end
    end

    k_iter = 2
    
    if k > k_iter
        cmblist_prev = similar(cmblist)
        cmblist_prev[:] .= cmblist[:]
        can_use_prev = can_use_phase2
        max_expl_prev = max_expl_phase2
    end

    prohib_old = prohib[2]
    num_prohibited_old = 0
    while k_iter < k
        k_iter += 1
        counter_loop = Int(0)
        cmblist_rec = zeros(Int64, pre_al, k_iter - 1)
        iter_cmb = Int(1)
        if k_iter == 3
            for i = 1 : length(can_use_prev)
                for j = 1 : numtar
                    if can_use_prev[i][j]
                        cmblist_rec[iter_cmb, :] .= [cmblist_prev[i, :] ; j]
                        iter_cmb += 1
                    end
                end
            end
        else
            c1, c2 = size(can_use_prev)
            for i = 1 : c1
                for j = 1 : c2
                    if can_use_prev[i, j]
                        cmblist_rec[iter_cmb, :] .= [cmblist_prev[i, :] ; j]
                        iter_cmb += 1
                    end
                end
            end
        end
        num_cmb = iter_cmb - 1

        prohib_mat = zeros(Int64, pre_al, k_iter)
        can_use_tmp = falses(num_cmb, numtar)
        max_expl_rec = zeros(Float64, pre_al)
        S = zeros(n, k_iter)
        info_bound = max(info_frac[k_iter] * (1 - ε), (1 - bestres) + (1 - bestres) * improvement)
        if doprint
            println("# of valid ", k_iter - 1, "-combinations : ", num_cmb)
            println("Checking combinations of size: ", k_iter)
            println("Maximum residual allowed: ", 1 - info_bound)
        end

        for j = 1 : num_cmb
            if doprint
                println("Combination# ", j)
            end

            for l = 1 : numtar
                idxisallowed = true
                for q = 1 : k_iter - 1
                    if l == cmblist_rec[j, q]
                        idxisallowed = false
                        break
                    end
                end

                flag = true
                idxcount = 0

                for ii = 1 : counter_loop
                    for jj = 1 : k_iter
                        for kk = 1 : k_iter - 1
                            if cmblist_rec[ii, kk] == prohib_mat[ii, jj]
                                idxcount += 1
                            end
                        end
                        if l == prohib_mat[ii, jj] && idxcount == k_iter - 1
                            flag = false
                            break
                        end
                    end
                end
                
                if idxisallowed && !can_use_tmp[j, l] && flag && (max_expl[l] + max_expl_prev[j]) >= info_bound
                    S[:, 1] .= A[:, cmblist_rec[j, 1]]
                    for q = 2 : k_iter - 1
                        S[:, q] .= A[:, cmblist_rec[j, q]]
                    end
                    S[:, k_iter] .= A[:, l]
                    tmp = ladFitGd_constrained_v2(S, b, σ, tol, δ, maxiter_gd)
                    tmpsol = zeros(numtar)
                    for q = 1 : k_iter - 1
                        tmpsol[cmblist_rec[j, q]] = tmp[q]
                    end
                    tmpsol[l] =  tmp[end]

                    tmpres = norm(b - A * tmpsol, 1) / norm(b, 1)

                    if tmpres < bestres
                        if doprint
                            for q = 1 : k_iter - 1
                                print(cmblist_rec[j, q], ", ")
                            end
                            println(l)
                            println("Residual = ", tmpres)
                        end
                        bestres = tmpres
                        bestsol .= tmpsol
                    end

                    if 1 - tmpres >= info_bound
                        can_use_tmp[j, l] = true
                        counter_loop += 1
                        if counter_loop > pre_al
                            pre_al *= 2
                            tmpprohib = prohib_mat
                            tmpmaxepl = max_expl_rec
                            prohib_mat = zeros(Int64, pre_al, k_iter)
                            max_expl_rec = zeros(Float64, pre_al)
                            prohib_mat[1:counter_loop - 1, :] .= tmpprohib
                            max_expl_rec[1:counter_loop - 1, :] .= tmpmaxepl
                        end
                        max_expl_rec[counter_loop] = 1 - tmpres
                        prohib_mat[counter_loop, 1:k_iter-1] .= cmblist_rec[j, :]
                        prohib_mat[counter_loop, end] = l
                    end
                else
                    if print_debug && cmblist_rec[j, end] == 219 && l == 307
                        for q = 1 : k_iter - 1
                            print(cmblist_rec[j, q], ", ")
                        end
                        println(l, ", excluded from search")
                        println("Is redundant? ", !idxisallowed)
                        println("Is prohibited? ", isprohib_v4([cmblist_rec[j, :] ; l], prohib_mat, k_iter, counter_loop))
                        println("Info Bound = ", info_bound)
                        println("Upper Bound = ", (1 - max_expl[l]))
                    end
                end

            end
        end
        cmblist_prev = cmblist_rec
        can_use_prev = can_use_tmp
        max_expl_prev = max_expl_rec
        prohib_old = prohib_mat
        num_prohibited_old = counter_loop
    end

    return bestsol, bestres
end;

function ladConstrained_1d(x, y, ε)
    coeff = y \ x
    cy = y * coeff
    r = x - cy

    if length(size(r)) > 1
        r = dropdims(r, dims=2)
    end

    sp = sortperm(r)

    return (x[sp[1]] - ε) / y[sp[1]]
end;

function ladFitGd_constrained_v2(A, b, ε, opts...)
    dim_list = size(A)
    if length(dim_list) > 1
        m = dim_list[1]
        n = dim_list[2]
    else
        m = dim_list[1]
        n = 1
    end
    if length(opts) == 4
        tol = opts[1]
        lambda = opts[2]
        maxiter = opts[3]
        doprint = opts[4]
    elseif length(opts) == 3
        tol = opts[1]
        lambda = opts[2]
        maxiter = opts[3]
        doprint = false
    elseif length(opts) == 2
        tol = opts[1]
        lambda = opts[2]
        maxiter = 1000
        doprint = false
    elseif length(opts) == 1
        tol = opts[1]
        lambda = 0.01
        maxiter = 1000
        doprint = false
    else
        tol = 0.001
        lambda = 0.01
        maxiter = 1000
        doprint = false
    end

    prev_step::Float64 = Float64(1)
    if length(dim_list) > 1
        cur_x = ones(size(A)[2], 1)
    else
        error("use ladConstrained_1d for 1 dimensional x")
    end
    δ = [.1, .1]
    
    if all(b .- A * cur_x .> ε) && all(cur_x .> 0)
        initres = norm(A*cur_x .- b, 1) + δ[1] * sum(-log.(cur_x)) + δ[2] * sum(-log.(b .- A * cur_x .- ε))
    else
        initres = Inf
    end
    if doprint
        println("initial guess = ", cur_x)
    end
    while isinf(initres)
        cur_x = cur_x .* .5
        
        if all(b .- A * cur_x .> ε) && all(cur_x .> 0)
            initres = norm(A*cur_x .- b, 1) + δ[1] * sum(-log.(cur_x)) + δ[2] * sum(-log.(b .- A * cur_x .- ε))
        else
            initres = Inf
        end
    end
    if doprint
        println("ε = ", ε)
        println("feasible starting point = ", cur_x)
        println("l1 norm = ", norm(A*cur_x .- b, 1))
        println("2nd term = ", δ[1] * sum(-log.(cur_x)))
        println("3rd term = ", δ[2] * sum(-log.(b .- A * cur_x .- ε)))
        println("initres = ", initres)
    end
    iter = 0
    lambda = .5
    pg = zeros(size(cur_x))
    t1 = zeros(size(cur_x))
    t2 = zeros(size(cur_x))
    t3 = zeros(size(cur_x))
    while iter < maxiter && prev_step > tol
        if all(b .- A * cur_x .> ε) && all(cur_x .> 0)
            res = norm(A*cur_x .- b, 1) + δ[1] * sum(-log.(cur_x)) + δ[2] * sum(-log.(b .- A * cur_x .- ε))
        else
            res = Inf
        end
        
        t1 .= transpose(A) * sign.(A * cur_x .- b)
        t2[:] .= 0
        if all(cur_x .> 0)
            t2 .= δ[1] * (-1 ./ cur_x)
        elseif !any(cur_x .< 0)
            for i = 1 : size(t2)[1]
                if cur_x[i] > 0
                    t2[i] = δ[1] * (-1 ./ cur_x[i])
                end
            end
        end
        t3[:] .= 0
        for i = 1 : size(A)[1]
            numer = A[i, :]
            denom = b[i] - dot(A[i, :], cur_x) - ε
            if denom > 0
                t3 .+= (numer ./ denom)
            end
        end
        g = t1 + t2 + (δ[2] * t3)
        if doprint
            println("res = ", res)
            println("t1 = ", t1)
            println("t2 = ", t2)
            println("t3 = ", t3)
            println("grad = ", g)
        end

        cur_x = cur_x - (g .* lambda)
        prev_step = norm(g, 2)

        iter = iter + 1
        normdiff = abs(norm(pg + g, 2))
        if doprint
            println("prev_step = ", prev_step)
            println("pg = ", pg)
            println("g = ", g)
            println("diff = ", pg + g)
            println("norm diff = ", normdiff)
            println("norm diff scaled = ", normdiff / lambda)
            println("iter = ", iter)
        end

        if normdiff < .125
             lambda *= .8
        end
        pg .= g
        if doprint
            println("lambda = ", lambda)
            println("cur_x = ", cur_x)
        end
    end
    
    return dropdims(cur_x, dims = 2)
end;

function isprohib_v4(idx, prohib, level, len)
    l = length(idx)
    if l == 1
        for i = 1 : len
            for j = 1 : level
                if idx == prohib[i, j]
                    return true
                end
            end
        end
    else
        for i = 1 : len
            counter = 0
            for j = 1 : level
                for k = 1 : l
                    if idx[k] == prohib[i, j]
                        counter += 1
                    end
                end
            end
            if counter == level
                return true
            end
        end
    end
    return false
end;
