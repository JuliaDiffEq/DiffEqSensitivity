@with_kw struct eFAST <: GSAMethod
    num_harmonics::Int=4
end

struct eFASTResult{T1}
    first_order::T1
    total_order::T1
end

function gsa(f,method::eFAST,p_range::AbstractVector;n::Int=1000,batch=false, kwargs...)
    @unpack num_harmonics = method
    num_params = length(p_range)
    omega = [floor((n-1)/(2*num_harmonics))]
    m = floor(omega[1]/(2*num_harmonics))
    desol = false
    
    if m>= num_params-1
        append!(omega,floor.(collect(range(1,stop=m,length=num_params-1))))
    else
        append!(omega,collect(range(0,stop=num_params-2)).%m .+1)
    end

    omega_temp = similar(omega)
    first_order = []
    total_order = []
    s = collect((2*pi/n) * (0:n-1))
    ps = zeros(num_params,n*num_params)

    for i in 1:num_params
        omega_temp[i] = omega[1]
        omega_temp[[k for k in 1:num_params if k != i]] = omega[2:end]
        l = collect((i-1)*n+1:i*n)
        phi = 2*pi*rand()
        for j in 1:num_params
            x =  0.5 .+ (1/pi) .*(asin.(sin.(omega_temp[j]*s .+ phi)))
            ps[j,l] .= quantile.(Uniform(p_range[j][1], p_range[j][2]),x)
        end
    end

    if batch
        all_y = f(ps)
        multioutput = all_y isa AbstractMatrix
    else
        _y = [f(ps[:,j]) for j in 1:size(ps,2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
            desol = true
        end
        all_y = multioutput ? reduce(hcat,_y) : _y
    end

    for i in 1:num_params
        if !multioutput
            ft = (fft(all_y[(i-1)*n+1:i*n]))[2:Int(floor((n/2)))]
            ys = ((abs.(ft))./n).^2 
            varnce = 2*sum(ys)
            push!(first_order,2*sum(ys[(1:num_harmonics)*Int(omega[1])])/varnce)
            push!(total_order,1 .- 2*sum(ys[1:Int(floor(omega[1]/2))])/varnce)
        else
            ft = [(fft(all_y[j,(i-1)*n+1:i*n]))[2:Int(floor((n/2)))] for j in 1:size(all_y,1)]
            ys = [((abs.(ff))./n).^2 for ff in ft]
            varnce = 2*sum.(ys)
            push!(first_order, map((y,var) -> 2*sum(y[(1:num_harmonics)*Int(omega[1])])./var, ys, varnce))
            push!(total_order, map((y,var) -> 1 .- 2*sum(y[1:Int(floor(omega[1]/2))])./var, ys, varnce))
        end
    end
    if desol 
        f_shape = x -> [reshape(x[:,i],y_size) for i in 1:size(x,2)]  
        first_order = map(f_shape,first_order)
        total_order = map(f_shape,total_order)
    end
    return eFASTResult(reduce(hcat,first_order), reduce(hcat,total_order))
end
