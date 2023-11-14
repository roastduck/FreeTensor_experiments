
#include <cpu_runtime.h>

extern "C" {
void run(void **_params, void **_returns, size_t **_retShapes, size_t *_retDims, CPUContext_t _ctx) {
  size_t _sharedStackSize = 439552;
  size_t _threadStackSize = 320;
  auto __stack = new uint8_t[_sharedStackSize + omp_get_max_threads() * _threadStackSize];
{
  auto &&_faces = mdspan_r<const float, extents<1, 5856, 3, 3>>((const float*)(_params[0]));
  auto &&_textures = mdspan_r<const float, extents<1, 5856, 25, 3>>((const float*)(_params[1]));
  auto &&_soft__colors = mdspan_r<float, extents<1, 4, 256, 256>>((float*)(_params[2]));
  auto &&_faces__inv = mdspan_r<float, extents<1, 5856, 3, 3>>((float*)(&__stack[0]));
  auto &&_faces__sym = mdspan_r<float, extents<1, 5856, 3, 3>>((float*)(&__stack[210816]));
  auto &&_faces__obt = mdspan_r<bool, extents<1, 5856, 3>>((bool*)(&__stack[421632]));
  for (int _fn = 0; _fn < 5856; _fn++) {
    {
      UncheckedOpt<mdspan_r<float, extents<3, 3>>> _inv_opt;
      auto &_inv = *_inv_opt;
      {
        float _det;
        _det = 0x0p+0f;
        for (int _p = 0; _p < 3; _p++) {
          _det += (_faces(0, _fn, _p, 0) * (_faces(0, _fn, ((_p + 1) % 3), 1) - _faces(0, _fn, ((_p + 2) % 3), 1)));
        }
        _det = ((_det > 0) ? std::max<float>(_det, 0x1.b7cdfep-34f) : std::min<float>(_det, -0x1.b7cdfep-34f));
        _inv_opt = mdspan_r<float, extents<3, 3>>((float*)(new float[(3)*(3)]));
        for (int _p__1 = 0; _p__1 < 3; _p__1++) {
          _inv(_p__1, 0) = ((_faces(0, _fn, ((_p__1 + 1) % 3), 1) - _faces(0, _fn, ((_p__1 + 2) % 3), 1)) / _det);
          _inv(_p__1, 1) = ((_faces(0, _fn, ((_p__1 + 2) % 3), 0) - _faces(0, _fn, ((_p__1 + 1) % 3), 0)) / _det);
          _inv(_p__1, 2) = (((_faces(0, _fn, ((_p__1 + 1) % 3), 0) * _faces(0, _fn, ((_p__1 + 2) % 3), 1)) - (_faces(0, _fn, ((_p__1 + 2) % 3), 0) * _faces(0, _fn, ((_p__1 + 1) % 3), 1))) / _det);
        }
      }
      for (int _i = 0; _i < 3; _i++) {
        for (int _i__1 = 0; _i__1 < 3; _i__1++) {
          _faces__inv(0, _fn, _i, _i__1) = _inv(_i, _i__1);
        }
      }
      auto _inv_ptr = _inv.data_handle();
      _inv_opt.drop();
      _inv_opt = std::nullopt;
      delete[] _inv_ptr;
    }
    for (int _i__2 = 0; _i__2 < 3; _i__2++) {
      for (int _i__3 = 0; _i__3 < 3; _i__3++) {
        _faces__sym(0, _fn, _i__2, _i__3) = (((_faces(0, _fn, _i__2, 0) * _faces(0, _fn, _i__3, 0)) + (_faces(0, _fn, _i__2, 1) * _faces(0, _fn, _i__3, 1))) + 1);
      }
    }
    {
      auto &&_obt = mdspan_r<bool, extents<3>>((bool*)(&__stack[439232]));
      for (int _k__1 = 0; _k__1 < 3; _k__1++) {
        auto &&_y = mdspan_r<float, extents<2>>((float*)(&__stack[439296]));
        for (int _k__2 = 0; _k__2 < 2; _k__2++) {
          _y(_k__2) = (_faces(0, _fn, ((_k__1 + 1) % 3), _k__2) - _faces(0, _fn, _k__1, _k__2));
        }
        {
          auto &&_y__1 = mdspan_r<float, extents<2>>((float*)(&__stack[439360]));
          for (int _k__3 = 0; _k__3 < 2; _k__3++) {
            _y__1(_k__3) = (_faces(0, _fn, ((_k__1 + 2) % 3), _k__3) - _faces(0, _fn, _k__1, _k__3));
          }
          _obt(_k__1) = (((_y(0) * _y__1(0)) + (_y(1) * _y__1(1))) < 0);
        }
      }
      for (int _i__4 = 0; _i__4 < 3; _i__4++) {
        _faces__obt(0, _fn, _i__4) = _obt(_i__4);
      }
    }
  }
  for (int _pn = 0; _pn < 65536; _pn++) {
    float _softmax__sum;
    _softmax__sum = 0x1.5829dcp+14f;
    {
      float _softmax__max;
      _softmax__max = 0x1.0624dep-10f;
      {
        auto &&_soft__color = mdspan_r<float, extents<4>>((float*)(&__stack[439232]));
        _soft__color(3) = 0x1p+0f;
        for (int _k__4 = 0; _k__4 < 3; _k__4++) {
          _soft__color(_k__4) = (_soft__colors(0, _k__4, (_pn / 256), (_pn % 256)) * 0x1.5829dcp+14f);
        }
        for (int _fn__1 = 0; _fn__1 < 5856; _fn__1++) {
          if (((((((((0x1p+1f * (_pn % 256)) + 0x1p+0f) - 256) / 256) <= (std::max<float>(std::max<float>(_faces(0, _fn__1, 0, 0), _faces(0, _fn__1, 1, 0)), _faces(0, _fn__1, 2, 0)) + 0x1.3a7978p-7f)) && (((((0x1p+1f * (_pn % 256)) + 0x1p+0f) - 256) / 256) >= (std::min<float>(std::min<float>(_faces(0, _fn__1, 0, 0), _faces(0, _fn__1, 1, 0)), _faces(0, _fn__1, 2, 0)) - 0x1.3a7978p-7f))) && (((((0x1p+1f * (((-1 * _pn) + 65535) / 256)) + 0x1p+0f) - 256) / 256) <= (std::max<float>(std::max<float>(_faces(0, _fn__1, 0, 1), _faces(0, _fn__1, 1, 1)), _faces(0, _fn__1, 2, 1)) + 0x1.3a7978p-7f))) && (((((0x1p+1f * (((-1 * _pn) + 65535) / 256)) + 0x1p+0f) - 256) / 256) >= (std::min<float>(std::min<float>(_faces(0, _fn__1, 0, 1), _faces(0, _fn__1, 1, 1)), _faces(0, _fn__1, 2, 1)) - 0x1.3a7978p-7f)))) {
            auto &&_w = mdspan_r<float, extents<3>>((float*)(&__stack[439296]));
            _w(0) = (((_faces__inv(0, _fn__1, 0, 0) * ((((0x1p+1f * (_pn % 256)) + 0x1p+0f) - 256) / 256)) + (_faces__inv(0, _fn__1, 0, 1) * ((((0x1p+1f * (((-1 * _pn) + 65535) / 256)) + 0x1p+0f) - 256) / 256))) + _faces__inv(0, _fn__1, 0, 2));
            _w(1) = (((_faces__inv(0, _fn__1, 1, 0) * ((((0x1p+1f * (_pn % 256)) + 0x1p+0f) - 256) / 256)) + (_faces__inv(0, _fn__1, 1, 1) * ((((0x1p+1f * (((-1 * _pn) + 65535) / 256)) + 0x1p+0f) - 256) / 256))) + _faces__inv(0, _fn__1, 1, 2));
            _w(2) = (((_faces__inv(0, _fn__1, 2, 0) * ((((0x1p+1f * (_pn % 256)) + 0x1p+0f) - 256) / 256)) + (_faces__inv(0, _fn__1, 2, 1) * ((((0x1p+1f * (((-1 * _pn) + 65535) / 256)) + 0x1p+0f) - 256) / 256))) + _faces__inv(0, _fn__1, 2, 2));
            {
              auto &&_sign = mdspan_r<float, extents<1>>((float*)(&__stack[439360]));
              if (((((((_w(0) > 0) && (_w(1) > 0)) && (_w(2) > 0)) && (_w(0) < 1)) && (_w(1) < 1)) && (_w(2) < 1))) {
                _sign(0) = 1;
              }
              else {
                _sign(0) = -1;
              }
              {
                float _dis__x;
                float _dis__y;
                if ((_sign(0) > 0)) {
                  float _dis__min;
                  _dis__min = 100000000;
                  {
                    float _dis__x__min;
                    _dis__x__min = 0;
                    {
                      float _dis__y__min;
                      _dis__y__min = 0;
                      {
                        auto &&_t0 = mdspan_r<float, extents<3>>((float*)(&__stack[439424]));
                        for (int _k__5 = 0; _k__5 < 3; _k__5++) {
                          {
                            auto &&_y__3 = mdspan_r<float, extents<3>>((float*)(&__stack[439488]));
                            for (int _k__6 = 0; _k__6 < 3; _k__6++) {
                              _y__3(_k__6) = (_faces__sym(0, _fn__1, _k__5, _k__6) - _faces__sym(0, _fn__1, ((_k__5 + 1) % 3), _k__6));
                            }
                            _t0(_k__5) = (((((_w(0) * _y__3(0)) + (_w(1) * _y__3(1))) + (_w(2) * _y__3(2))) - _y__3(((_k__5 + 1) % 3))) / (_y__3(_k__5) - _y__3(((_k__5 + 1) % 3))));
                          }
                          _t0(((_k__5 + 1) % 3)) = (1 - _t0(_k__5));
                          _t0(((_k__5 + 2) % 3)) = 0;
                          for (int _i__5 = 0; _i__5 < 3; _i__5++) {
                            _t0(_i__5) -= _w(_i__5);
                          }
                          _dis__x = (((_t0(0) * _faces(0, _fn__1, 0, 0)) + (_t0(1) * _faces(0, _fn__1, 1, 0))) + (_t0(2) * _faces(0, _fn__1, 2, 0)));
                          _dis__y = (((_t0(0) * _faces(0, _fn__1, 0, 1)) + (_t0(1) * _faces(0, _fn__1, 1, 1))) + (_t0(2) * _faces(0, _fn__1, 2, 1)));
                          {
                            float _dis;
                            _dis = (runtime_square(_dis__x) + runtime_square(_dis__y));
                            if ((_dis < _dis__min)) {
                              _dis__min = _dis;
                              _dis__x__min = _dis__x;
                              _dis__y__min = _dis__y;
                            }
                          }
                        }
                      }
                      _dis__x = _dis__x__min;
                      _dis__y = _dis__y__min;
                    }
                  }
                }
                else {
                  UncheckedOpt<mdspan_r<float, extents<3>>> _t_opt;
                  auto &_t = *_t_opt;
                  {
                    int32_t _v0;
                    _v0 = -1;
                    for (int _k__7 = 0; _k__7 < 3; _k__7++) {
                      if ((((_v0 == -1) && (_w(((_k__7 + 1) % 3)) <= 0)) && (_w(((_k__7 + 2) % 3)) <= 0))) {
                        _v0 = _k__7;
                        if ((_faces__obt(0, _fn__1, _k__7) && ((((((((0x1p+1f * (_pn % 256)) + 0x1p+0f) - 256) / 256) - _faces(0, _fn__1, _k__7, 0)) * (_faces(0, _fn__1, ((_k__7 + 2) % 3), 0) - _faces(0, _fn__1, _k__7, 0))) + ((((((0x1p+1f * (((-1 * _pn) + 65535) / 256)) + 0x1p+0f) - 256) / 256) - _faces(0, _fn__1, _k__7, 1)) * (_faces(0, _fn__1, ((_k__7 + 2) % 3), 1) - _faces(0, _fn__1, _k__7, 1)))) > 0))) {
                          _v0 = ((_k__7 + 2) % 3);
                        }
                      }
                    }
                    for (int _k__8 = 0; _k__8 < 3; _k__8++) {
                      if (((_v0 == -1) && (_w(_k__8) <= 0))) {
                        _v0 = ((_k__8 + 1) % 3);
                      }
                    }
                    {
                      auto &&_y__4 = mdspan_r<float, extents<3>>((float*)(&__stack[439424]));
                      for (int _k__9 = 0; _k__9 < 3; _k__9++) {
                        _y__4(_k__9) = (_faces__sym(0, _fn__1, _v0, _k__9) - _faces__sym(0, _fn__1, runtime_mod((_v0 + 1), 3), _k__9));
                      }
                      _t_opt = mdspan_r<float, extents<3>>((float*)(new float[(3)]));
                      _t(_v0) = (((((_w(0) * _y__4(0)) + (_w(1) * _y__4(1))) + (_w(2) * _y__4(2))) - _y__4(runtime_mod((_v0 + 1), 3))) / (_y__4(_v0) - _y__4(runtime_mod((_v0 + 1), 3))));
                    }
                    _t(runtime_mod((_v0 + 1), 3)) = (1 - _t(_v0));
                    _t(runtime_mod((_v0 + 2), 3)) = 0;
                  }
                  for (int _k__10 = 0; _k__10 < 3; _k__10++) {
                    _t(_k__10) = (std::min<float>(std::max<float>(_t(_k__10), 0x0p+0f), 0x1p+0f) - _w(_k__10));
                  }
                  _dis__x = (((_t(0) * _faces(0, _fn__1, 0, 0)) + (_t(1) * _faces(0, _fn__1, 1, 0))) + (_t(2) * _faces(0, _fn__1, 2, 0)));
                  _dis__y = (((_t(0) * _faces(0, _fn__1, 0, 1)) + (_t(1) * _faces(0, _fn__1, 1, 1))) + (_t(2) * _faces(0, _fn__1, 2, 1)));
                  auto _t_ptr = _t.data_handle();
                  _t_opt.drop();
                  _t_opt = std::nullopt;
                  delete[] _t_ptr;
                }
                {
                  float _dis__1;
                  _dis__1 = (runtime_square(_dis__x) + runtime_square(_dis__y));
                  if (((_sign(0) >= 0) || (_dis__1 < 0x1.824e34p-14f))) {
                    float _soft__fragment;
                    _soft__fragment = (0x1p+0f / (0x1p+0f + exp((((-1 * _sign(0)) * _dis__1) / 0x1.4f8b58p-17f))));
                    _soft__color(3) *= (0x1p+0f - _soft__fragment);
                    {
                      //auto &&_w__clip = mdspan_r<float, extents<3>>((float*)(&__stack[439424]));
                      auto &&_w__clip = mdspan_r<float, extents<3>>((float*)(&__stack[439232 + omp_get_thread_num() * _threadStackSize + 192]));
                      for (int _k__11 = 0; _k__11 < 3; _k__11++) {
                        _w__clip(_k__11) = std::max<float>(std::min<float>(_w(_k__11), 0x1p+0f), 0x0p+0f);
                      }
                      {
                        float _w__sum;
                        _w__sum = std::max<float>(((_w__clip(0) + _w__clip(1)) + _w__clip(2)), 0x1.4f8b58p-17f);
                        for (int _k__12 = 0; _k__12 < 3; _k__12++) {
                          _w__clip(_k__12) = (_w__clip(_k__12) / _w__sum);
                        }
                      }
                      {
                        float _zp;
                        _zp = (0x1p+0f / (((_w__clip(0) / _faces(0, _fn__1, 0, 2)) + (_w__clip(1) / _faces(0, _fn__1, 1, 2))) + (_w__clip(2) / _faces(0, _fn__1, 2, 2))));
                        if (((_zp >= 0x1p+0f) && (_zp <= 0x1.9p+6f))) {
                          if ((((_faces(0, _fn__1, 2, 1) - _faces(0, _fn__1, 0, 1)) * (_faces(0, _fn__1, 1, 0) - _faces(0, _fn__1, 0, 0))) < ((_faces(0, _fn__1, 1, 1) - _faces(0, _fn__1, 0, 1)) * (_faces(0, _fn__1, 2, 0) - _faces(0, _fn__1, 0, 0))))) {
                            float _zp__norm;
                            _zp__norm = ((0x1.9p+6f - _zp) / 0x1.8cp+6f);
                            {
                              float _exp__delta__zp;
                              _exp__delta__zp = 0x1p+0f;
                              if ((_zp__norm > _softmax__max)) {
                                _exp__delta__zp = exp(((_softmax__max - _zp__norm) / 0x1.a36e2ep-14f));
                                _softmax__max = _zp__norm;
                              }
                              _softmax__sum = ((_exp__delta__zp * _softmax__sum) + (exp(((_zp__norm - _softmax__max) / 0x1.a36e2ep-14f)) * _soft__fragment));
                              for (int _k__13 = 0; _k__13 < 3; _k__13++) {
                                float _texture__k;
                                {
                                  int32_t _w__x;
                                  _w__x = int32_t((_w__clip(0) * 5));
                                  {
                                    int32_t _w__y;
                                    _w__y = int32_t((_w__clip(1) * 5));
                                    if ((((((_w__clip(0) + _w__clip(1)) * 5) - _w__x) - _w__y) <= 1)) {
                                      if ((((_w__y * 5) + _w__x) == 25)) {
                                        _texture__k = _textures(0, _fn__1, ((-1 * runtime_mod((4 * _w__x), 5)) + 24), _k__13);
                                      }
                                      else {
                                        _texture__k = _textures(0, _fn__1, ((_w__y * 5) + _w__x), _k__13);
                                      }
                                    }
                                    else {
                                      _texture__k = _textures(0, _fn__1, (((-1 * _w__x) + (-5 * _w__y)) + 24), _k__13);
                                    }
                                  }
                                }
                                _soft__color(_k__13) = ((_exp__delta__zp * _soft__color(_k__13)) + ((exp(((_zp__norm - _softmax__max) / 0x1.a36e2ep-14f)) * _soft__fragment) * _texture__k));
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        _soft__colors(0, 3, (_pn / 256), (_pn % 256)) = (0x1p+0f - _soft__color(3));
        for (int _k__14 = 0; _k__14 < 3; _k__14++) {
          _soft__colors(0, _k__14, (_pn / 256), (_pn % 256)) = (_soft__color(_k__14) / _softmax__sum);
        }
      }
    }
  }
}
  delete[] __stack;
}
}
