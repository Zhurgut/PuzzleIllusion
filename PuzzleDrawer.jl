
using LinearAlgebra: norm
X = 0:0.01:1

function xt(f)
    d = 1e-6
    df(x) = (f(x+d) - f(x)) / d
    xs = []
    x = 0.0
    v = 0.01

    while x <= 1
        push!(xs, x)
        x += v / sqrt(1 + df(x)^2)
    end

    return xs
end

function envelope(x)
    if x < 0.25
        return x
    elseif x < 0.75
        return 0.25
    else
        return 1-x
    end
end

runge(x) = 1 / (1+x^2)
drunge(x) = -2x / (1+x^2)^2

f1(x) = (runge(8x-4) - 1/17) / (1-1/17)
f2(x) = -4 * x * (x - 1)
f3(x) = 0.35 <= x <= 0.65 ? 1.0 : 0.0
beta1(x) = x^4*(1-x)^6
beta2(x) = x^6.5*(1-x)^4
f4(x) = beta1(x) / maximum(beta1.(0:0.01:1))
f5(x) = beta2(x) / maximum(beta2.(0:0.01:1))
f6(x) = x < 0.5 ? 2x : 2-2x
f7(x) = cos(3(x-0.5)π) / exp(10(x-0.5)^2)

X = 0:0.01:1
plot(X, envelope.(X) .* f1.(X), ratio=1)
plot(X, envelope.(X) .* f2.(X), ratio=1)
plot(X, envelope.(X) .* f3.(X), ratio=1)
plot(X, envelope.(X) .* f4.(X), ratio=1)
plot(X, envelope.(X) .* f5.(X), ratio=1)
plot(X, envelope.(X) .* f6.(X), ratio=1)
plot(X, envelope.(X) .* f7.(X), ratio=1)

function plot_random_combo()
    f = [f1, f3, f4, f5, f6, f7]
    r = (randn(length(f))).^2
    r .*= (1 / sum(r))
    display(r)
    X = 0:0.01:1
    foo(x) = sum([r[i] * f[i](x) for i in eachindex(f)])
    plot(X, envelope.(X) .* foo.(X), ratio=1)
end

knob(x) = 4(x-0.5)^2 * sin(2x*π)
knob2(x) = (1-4(x-0.5)^2) * 4(x-0.5)^2 * sin(2x*π)

function plot_connector(f, ;kf=knob, w = 0.5)
    x = xt(f)
    points = [(x[t] + w*kf(x[t]), envelope(x[t]) * f(x[t])) for t in eachindex(x)]
    scatter([p[1] for p in points], [p[2] for p in points], ratio=1)
end

function connector_function(f, knob_fn, w)
    x = xt(f)
    function c(t)
        i = round(Int, t*(length(x)-1))+1
        return [x[i] + w*knob_fn(x[i]), envelope(x[i]) * f(x[i])]
    end
end

function random_connector()
    f = [f1, f2, f3, f4, f5, f6, f7]
    w = [0.7, 0.7, 0.6, 0.6, 0.6, 0.7, 0.5]
    kf = [knob, knob, knob, knob, knob, knob, knob2]
    r = (randn(length(f))).^2
    r .*= (1 / sum(r))
    cs = [connector_function(f[i], kf[i], w[i]) for i in eachindex(f)]
    return x->(sum((r[i] * cs[i](x) for i in eachindex(cs))))

end

function plot_connector(f)
    X = 0:0.01:1
    out = f.(X)
    scatter([o[1] for o in out], [o[2] for o in out], ratio=1)
end