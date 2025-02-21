#include <Eigen/Core>
#include <math/types.hpp>
#include <math/bounding_box.hpp>
#include <vector>
#include <exception>
#include <stdexcept>

namespace spurt {

class symplectic4D {
public:
    typedef spurt::small_vector<unsigned long int, 4> ulivec4;
    typedef vec4 state_type;
    typedef mat4 deriv_type;
    typedef spurt::bounding_box<state_type> bounds_type;

    static constexpr double onepi = 3.1415926535897932384626433;
    static constexpr double twopi = 6.2831853071795864769252868;
    double k1, k2, eps;

private:
    void forward(double& p1, double& p2, double& q1, double& q2) const {
        q1 += p1;
        q2 += p2;
        p1 += k1/twopi * sin(twopi * q1) + eps/twopi * sin(twopi * (q1 + q2));
        p2 += k2/twopi * sin(twopi * q2) + eps/twopi * sin(twopi * (q1 + q2));
    }

    void backward(double& p1, double& p2, double& q1, double& q2) const {
        p1 -= k1/twopi * sin(twopi * q1) + eps/twopi * sin(twopi *(q1 + q2));
        p2 -= k2/twopi * sin(twopi * q2) + eps/twopi * sin(twopi *(q1 + q2));
        q1 -= p1;
        q2 -= p2;
    }

    deriv_type forwardJ(double p1, double p2, double q1, double q2) const {
        q1 += p1;
        q2 += p2;
        deriv_type J = 0;
        J(0,0) = 1; // dp1'/dp1
        J(0,2) = k1 * cos(twopi * q1) + eps * cos(twopi * (q1 + q2)); // dp1'/dq1
        J(0,3) = J(1,2) = eps*cos(twopi*(q1+q2)); // dp1'/dq2 = dp2'/dq1
        J(1,1) = 1; // dp2'/dp2
        J(1,3) = k2 * cos(twopi * q2) + eps * cos(twopi * (q1 + q2)); // dp2'/dq2
        J(2,0) = J(2,2) = 1; // dq1'/dp1 = dq1'/dq1
        J(3,1) = J(3,3) = 1; // dq2'/dp2 = dq2'/dq2
        return J;
    }

    deriv_type backwardJ(double p1, double p2, double q1, double q2) const {
        deriv_type J = 0;
        J(0,0) = 1; // dp1/dp1'
        J(0,2) = -k1 * cos(twopi * q1) - eps * cos(twopi * (q1 + q2)); // dp1/dq1'
        J(0,3) = J(1,2) = -eps * cos(twopi*( q1 + q2)); // dp1/dq2' = dp2/dq1'
        J(1,1) = 1; // dp2/dq1'
        J(1,3) = -k2 * cos(twopi * q2) - eps * cos(twopi * (q1 + q2)); // dp2/dq2'
        J(2,0) = -1;
        J(2,2) = 1 + k1 * cos(twopi * q1) + eps * cos(twopi * (q1 + q2)); // dq1/dq1'
        J(2,3) = J(3,2) = eps * cos(twopi * (q1 + q2)); // dq1/dq2' = dq2/dq1'
        J(3,1) = -1;
        J(3,3) = 1 + k2 * cos(twopi * q2) + eps * cos(twopi * (q1 + q2));  // dq2/dq2'
        return J;
    }

public:

    static void to_range(double& x) {
        if (x < -0.5 || x >= 0.5) {
            x -= std::floor(x);
            if (x >= 0.5) x -= 1;
            if (x < -0.5 || x >= 0.5) {
                std::cerr << "ERROR in to_range: " << x << std::endl;
            }
        }
    }

    static void to_range(double& x, unsigned long int& rot_ct) {
        if (x < -0.5 || x >= 0.5) {
            x -= std::floor(x);
            ++rot_ct;
            if (x >= 0.5) x -= 1;
            if (x < -0.5 || x >= 0.5) {
                std::cerr << "ERROR in to_range: " << x << std::endl;
            }
        }
    }

    static void to_domain(state_type& x) {
        for (int i=0; i<4; ++i) {
            to_range(x[i]);
        }
    }

    static void to_domain(state_type& x, ulivec4& rot_cts) {
        for (int i=0; i<4; ++i) {
            to_range(x[i], rot_cts[i]);
        }
    }

    struct map_undefined : public std::runtime_error {
        map_undefined() : std::runtime_error("undefined map at this location") {}
    };

    symplectic4D(double k1, double k2, double eps) : k1(k1), k2(k2), eps(eps) {}

    state_type map(const state_type& x, ulivec4& rot_cts,
                   int n = 1) const {
        state_type y(x);

        if (n>0) {
            for (int i=0; i<n; ++i) {
                forward(y[0], y[1], y[2], y[3]);
                to_domain(y, rot_cts);
            }
        }
        else if (n<0) {
            for (int i=n; i<0; ++i) {
                backward(y[0], y[1], y[2], y[3]);
                to_domain(y, rot_cts);
            }
        }
        return y;
    }

    void map(const state_type& x, ulivec4& rot_cts,
             std::vector< state_type >& hits, int n = 1) const {
        hits.resize(std::abs<int>(n));
        state_type y(x);
        if (n>0) {
            for (int i=0; i<n; ++i) {
                forward(y[0], y[1], y[2], y[3]);
                to_domain(y, rot_cts);
                hits[i] = y;
            }
        }
        else if (n<0) {
            for (int i=n; i<0; ++i) {
                backward(y[0], y[1], y[2], y[3]);
                to_domain(y, rot_cts);
                hits[i] = y;
            }
        }
    }

    void map(const state_type& x, ulivec4& rot_cts,
             std::vector<std::pair<state_type, deriv_type> >& out, int niter,
             double eps=0) const {
        out.resize(std::abs<int>(niter));
        state_type y = x;
        if (niter > 0) {
            for (int i=0; i<niter; ++i) {
                deriv_type J = forwardJ(y[0], y[1], y[2], y[3]);
                if (i>0) out[i].second = out[i-1].second * J;
                else out[i].second = J;
                forward(y[0], y[1], y[2], y[3]);
                to_domain(y, rot_cts);
                out[i].first = y;
            }
        }
        else if (niter < 0) {
            for (int i=niter; i<0; ++i) {
                deriv_type J = backwardJ(y[0], y[1], y[2], y[3]);
                if (i>niter) out[i].second = out[i-1].second * J;
                else out[i].second = J;
                backward(y[0], y[1], y[2], y[3]);
                to_domain(y, rot_cts);
                out[i].first = y;
            }
        }
    }

    std::pair<state_type, deriv_type>
    map_and_jacobian(const state_type& in, ulivec4& rot_cts, int niter, double eps=0) const {
        std::vector<std::pair<state_type, deriv_type> > out;
        map(in, rot_cts, out, niter);
        if (out.size() != std::abs(niter)) {
            throw map_undefined();
        }
        return out.back();
    }

    const bounds_type bounds() const {
        return bounds_type(state_type(-0.5,-0.5,-0.5,-0.5), state_type(0.5,0.5,0.5,0.5));
    }

    symplectic4D* clone() const {
        return new symplectic4D(*this);
    }

    double precision() const {
        return 0;
    }
    void precision(double) {}
};

class slab_map {
public:
    double thickness;
    int section_dim;
    typedef spurt::small_vector<unsigned long int, 4> ulivec4;
    typedef vec3 point_type;
    typedef vec4 state_type;
    ulivec4 rot_cts;

    slab_map(int _dim=3, double _thickness=1.0e-4)
        : thickness(_thickness), section_dim(_dim) {}

    template<typename Map>
    // TODO: pass rot_ct here, sync issues when in slab class
    void run(std::vector<state_type>& hits, std::vector<state_type>& orbit,
             const Map& map, const state_type& seed, int max_iter=1000,
             int nhits=-1) {
        hits.clear();
        orbit.clear();
        hits.push_back(seed);
        orbit.push_back(seed);
        state_type s = seed;
        map.to_domain(s);
        for (int i=0; i<4; ++i) {
            rot_cts[i] = 0;
        }
        s = _run(hits, orbit, map, s, rot_cts, max_iter);
    }

private:
    template<typename Map>
    state_type _run(std::vector<state_type>& hits,
                    std::vector<state_type>& orbit, const Map& map,
                    const state_type& seed, ulivec4& rot_cts, int niter) {
        map.map(seed, rot_cts, orbit, niter);
        for (int i=0; i<orbit.size(); ++i) {
            if (std::abs(orbit[i][section_dim]) <= thickness) {
                hits.push_back(orbit[i]);
            }
        }
        return orbit.back();
    }
};

} // namespace spurt
