Module tapenade_render
    implicit none

    ! Macro Definitions
    integer, parameter :: batch_size = 1
    integer, parameter :: num_faces = 5856
    integer, parameter :: texture_size = 25
    integer, parameter :: texture_res = 5
    real*4, parameter :: near = 1.0
    real*4, parameter :: far = 100.0
    real*4, parameter :: eps = 1e-3
    real*4, parameter :: sigma_val = 1e-5
    real*4, parameter :: gamma_val = 1e-4
    real*4, parameter :: dist_eps = 9.21024036697585
    real*4, parameter :: threshold = dist_eps * sigma_val
    logical, parameter :: double_side = .false.
    integer, parameter :: texture_type = 0
    
contains
    ! Function and Subroutine Declarations
    integer function mod_index(a, p)
        implicit none
        integer, intent(in) :: a, p
        mod_index = mod(a, p) + 1
    end function mod_index

    subroutine face_inv(inv_t, faces_t, fn, bn)
        implicit none
        real*4, intent(in) :: faces_t(3 * 3 * num_faces * batch_size)
        integer, intent(in) :: fn, bn
        real*4, intent(out) :: inv_t(3 * 3)
        real*4 :: det
        integer :: p
        integer :: d
        d = (bn - 1) * (num_faces * 3 * 3) + (fn - 1) * (3 * 3) + 1
        det = faces_t(d + 0) * (faces_t(d + 4) - faces_t(d + 7)) + &
              faces_t(d + 3) * (faces_t(d + 7) - faces_t(d + 1)) + &
              faces_t(d + 6) * (faces_t(d + 1) - faces_t(d + 4))
        
        if (det > 0) then
            det = max(det, 1e-10)
        else
            det = min(det, -1e-10)
        end if

        inv_t(1) = (faces_t(d + 4) - faces_t(d + 7)) / det;
        inv_t(2) = (faces_t(d + 6) - faces_t(d + 3)) / det;
        inv_t(3) = (faces_t(d + 3) * faces_t(d + 7) - faces_t(d + 6) * faces_t(d + 4)) / det;

        inv_t(4) = (faces_t(d + 7) - faces_t(d + 1)) / det;
        inv_t(5) = (faces_t(d + 0) - faces_t(d + 6)) / det;
        inv_t(6) = (faces_t(d + 6) * faces_t(d + 1) - faces_t(d + 0) * faces_t(d + 7)) / det;

        inv_t(7) = (faces_t(d + 1) - faces_t(d + 4)) / det;
        inv_t(8) = (faces_t(d + 3) - faces_t(d + 0)) / det;
        inv_t(9) = (faces_t(d + 0) * faces_t(d + 4) - faces_t(d + 3) * faces_t(d + 1)) / det;

    end subroutine face_inv
    
    real*4 function dot_xy(v1, v2)
        implicit none
        real*4, intent(in) :: v1(3)
        real*4, intent(in) :: v2(3)
        
        dot_xy = v1(1) * v2(1) + v1(2) * v2(2)
    end function dot_xy
    
    real*4 function cross_xy(v1, v2)
        implicit none
        real*4, intent(in) :: v1(3)
        real*4, intent(in) :: v2(3)
        
        cross_xy = v1(1) * v2(2) - v1(2) * v2(1)
    end function cross_xy
    
    subroutine sub_xy(v, v1, v2)
        implicit none
        real*4, intent(out) :: v(3)
        real*4, intent(in) :: v1(3)
        real*4, intent(in) :: v2(3)
        integer :: k
        
        do k = 1, 2
            v(k) = v1(k) - v2(k)
        end do
    end subroutine sub_xy
    
    real*4 function norm2(v)
        implicit none
        real*4, intent(in) :: v(3)
        
        norm2 = dot_xy(v, v)
    end function norm2
    
    subroutine barycentric_coordinate(w, p, inv)
        implicit none
        real*4, intent(out) :: w(3)
        real*4, intent(in) :: p(3)
        real*4, intent(in) :: inv(3 * 3)
        integer :: k
        w(1) = inv(1) * p(1) + inv(2) * p(2) + inv(3);
        w(2) = inv(4) * p(1) + inv(5) * p(2) + inv(6);
        w(3) = inv(7) * p(1) + inv(8) * p(2) + inv(9);
        
    end subroutine barycentric_coordinate
    
    logical function check_border(p, faces_t, threshold_t, fn, bn)
        real(4), intent(in) :: p(3), faces_t(3 * 3 * num_faces * batch_size)
        real(4), intent(in) :: threshold_t
        integer, intent(in) :: fn, bn
        integer :: d
        real(4) :: t
        d = (bn - 1) * (num_faces * 3 * 3) + (fn - 1) * (3 * 3) + 1
        t = sqrt(threshold_t)
        check_border = (p(1) > MAX(MAX(faces_t(d + 0), faces_t(d + 3)), faces_t(d + 6)) + t) .or. &
                       (p(1) < MIN(MIN(faces_t(d + 0), faces_t(d + 3)), faces_t(d + 6)) - t) .or. &
                       (p(2) > MAX(MAX(faces_t(d + 1), faces_t(d + 4)), faces_t(d + 7)) + t) .or. &
                       (p(2) < MIN(MIN(faces_t(d + 1), faces_t(d + 4)), faces_t(d + 7)) - t)
    end function check_border

    logical function check_face_frontside(faces_t, fn, bn)
        real(4), intent(in) :: faces_t(3 * 3 * num_faces * batch_size)
        integer, intent(in) :: fn, bn
        integer :: d
        d = (bn - 1) * (num_faces * 3 * 3) + (fn - 1) * (3 * 3) + 1

        check_face_frontside = (faces_t(d + 7) - faces_t(d + 1)) * (faces_t(d + 3) - faces_t(d + 0)) &
            < (faces_t(d + 4) - faces_t(d + 1)) * (faces_t(d + 6) - faces_t(d + 0))
    end function check_face_frontside

    subroutine barycentric_clip(w_clip, w)
        real(4), intent(in) :: w(3)
        real(4), intent(out) :: w_clip(3)
        real(4) :: w_sum
        integer :: k
        do k = 1, 3
            w_clip(k) = MAX(MIN(w(k), 1.0), 0.0)
        end do
        w_sum = MAX(w_clip(1) + w_clip(2) + w_clip(3), 1e-5)
        do k = 1, 3
            w_clip(k) = w_clip(k) / w_sum
        end do
    end subroutine barycentric_clip

    real(4) function euclidean_p2f_distance(faces_t, p, fn, bn)
        real(4), intent(in) :: faces_t(3 * 3 * num_faces * batch_size), p(3)
        integer, intent(in) :: fn, bn
        real(4) :: dis(3)
        real(4), dimension(3) :: tmp0, tmp1, t1, t2, t3
        real(4) :: area, d1, d2, len
        integer :: k, l
        integer :: d
        d = (bn - 1) * (num_faces * 3 * 3) + (fn - 1) * (3 * 3) + 1

        do k = 1, 3
            tmp0(1) = faces_t(d + (k - 1) * 3 + 0)
            tmp0(2) = faces_t(d + (k - 1) * 3 + 1)
            tmp0(3) = faces_t(d + (k - 1) * 3 + 2)

            tmp1(1) = faces_t(d + mod(k, 3) * 3 + 0)
            tmp1(2) = faces_t(d + mod(k, 3) * 3 + 1)
            tmp1(3) = faces_t(d + mod(k, 3) * 3 + 2)
            call sub_xy(t1, p, tmp0)
            call sub_xy(t2, tmp1, tmp0)
            area = cross_xy(t1, t2)
            d1 = dot_xy(t1, t2)
            if (d1 >= 0.0) then
                call sub_xy(t3, tmp1, p)
                d2 = dot_xy(t2, t3)
                if (d2 >= 0.0) then
                    len = norm2(t2)
                    dis(k) = area / MAX(len, 1e-10) * area
                else
                    dis(k) = norm2(t3)
                end if
            else
                dis(k) = norm2(t1)
            end if
        end do
        euclidean_p2f_distance = MIN(MIN(dis(1), dis(2)), dis(3))
    end function euclidean_p2f_distance

    real(4) function forward_sample_texture(textures_t, w, r, k, ty, fn, bn)
        real(4), intent(in) :: textures_t(3 * texture_size * num_faces * batch_size), w(3)
        integer, intent(in) :: r, k, ty, fn, bn
        integer :: w_x, w_y
        integer :: d
        real(4) :: texture_k
        d = (bn - 1) * (num_faces * texture_size * 3) + (fn - 1) * (texture_size * 3)
        if (ty == 0) then
            w_x = int(w(1) * r)
            w_y = int(w(2) * r)
            if ((w(1) + w(2)) * r - w_x - w_y <= 1) then
                if (w_y * r + w_x == texture_size) then
                    texture_k = textures_t(d + (w_y * r + w_x - 1) * 3 + k)
                else
                    texture_k = textures_t(d + (w_y * r + w_x) * 3 + k)
                end if
            else
                texture_k = textures_t(d + ((r - 1 - w_y) * r + (r - 1 - w_x)) * 3 + k)
            end if
        end if
        forward_sample_texture = texture_k
    end function forward_sample_texture

    subroutine tapenade_render_main(soft_colors_t, faces_t, textures_t) BIND(c)
        real(4), intent(inout) :: soft_colors_t(image_size * image_size * 4 * batch_size)
        real(4), intent(in) :: faces_t(3 * 3 * num_faces * batch_size)
        real(4), intent(in) :: textures_t(3 * texture_size * num_faces * batch_size)
        integer :: bn, pn, yi, xi, fn, k, d
        real(4), dimension(3) :: pixel
        real(4), dimension(3 * 3) :: inv
        real(4), dimension(3) :: w, w_clip
        real(4), dimension(3) :: soft_color
        real(4), dimension(num_faces) :: soft_color_alpha
        real(4) :: sign, softmax_max, softmax_sum, zp, zp_norm, coef, color_k, soft_fragment, dis
        do bn = 1, batch_size
            !$OMP PARALLEL DO SHARED(soft_colors_t, faces_t, textures_t, bn) PRIVATE(pn,&
            !$OMP&yi, xi, fn, k, d, pixel, inv, w, w_clip, soft_color, soft_color_alpha, sign, &
            !$OMP&softmax_max, softmax_sum, zp, zp_norm, coef, color_k, soft_fragment, dis)
            do pn = 1, image_size * image_size
                yi = image_size - 1 - int((pn - 1) / image_size)
                xi = mod((pn - 1), image_size)
                pixel(1) = (2.0 * xi + 1.0 - image_size) / image_size
                pixel(2) = (2.0 * yi + 1.0 - image_size) / image_size
                softmax_max = eps
                do k = 1, 3
                    soft_color(k) = 0.0
                end do
                soft_color_alpha(1) = 1.0
                do fn = 1, num_faces
                    d = (bn - 1) * (num_faces * 3 * 3) + (fn - 1) * (3 * 3) + 1
                    call face_inv(inv, faces_t, fn, bn)

                    if (.not. check_border(pixel, faces_t, threshold, fn, bn)) then
                        call barycentric_coordinate(w, pixel, inv)
                        call barycentric_clip(w_clip, w)

                        zp = 1.0 / (w_clip(1) / faces_t(d + 2) + w_clip(2) / faces_t(d + 5) + &
                            w_clip(3) / faces_t(d + 8))

                        if (.not. (zp < near .or. zp > far)) then
                            if (check_face_frontside(faces_t, fn, bn) .or. double_side) then
                                zp_norm = (far - zp) / (far - near)

                                if (zp_norm > softmax_max) then
                                    softmax_max = zp_norm
                                end if
                            end if
                        end if
                    end if
                end do

                softmax_sum = exp((2 * eps - softmax_max) / gamma_val)

                do k = 1, 3
                    soft_color(k) = soft_colors_t((bn - 1) * (4 * image_size * image_size) + &
                        (k - 1) * (image_size * image_size) + pn) * softmax_sum
                end do

                do fn = 1, num_faces
                    d = (bn - 1) * (num_faces * 3 * 3) + (fn - 1) * (3 * 3) + 1
                    if (fn > 1) then 
                        soft_color_alpha(fn) = soft_color_alpha(fn - 1)
                    end if 

                    call face_inv(inv, faces_t, fn, bn)

                    if (.not. check_border(pixel, faces_t, threshold, fn, bn)) then
                        call barycentric_coordinate(w, pixel, inv)

                        if (w(1) > 0 .and. w(2) > 0 .and. w(3) > 0 .and. w(1) < 1 .and. w(2) < 1 .and. w(3) < 1) then
                            sign = 1.0
                        else
                            sign = -1.0
                        end if

                        dis = euclidean_p2f_distance(faces_t, pixel, fn, bn)

                        if (.not. (sign < 0.0 .and. dis >= threshold)) then
                            soft_fragment = 1.0 / (1.0 + exp(-sign * dis / sigma_val))

                            if (fn > 1) then
                                soft_color_alpha(fn) = soft_color_alpha(fn - 1) * (1.0 - soft_fragment)
                            else 
                                soft_color_alpha(fn) = (1.0 - soft_fragment)
                            end if

                            call barycentric_clip(w_clip, w)

                            zp = 1.0 / (w_clip(1) / faces_t(d + 2) + w_clip(2) / faces_t(d + 5) + &
                                w_clip(3) / faces_t(d + 8))

                            if (.not. (zp < near .or. zp > far)) then
                                if (check_face_frontside(faces_t, fn, bn) .or. double_side) then
                                    zp_norm = (far - zp) / (far - near)

                                    coef = exp((zp_norm - softmax_max) / gamma_val) * soft_fragment
                                    softmax_sum = softmax_sum + coef

                                    do k = 1, 3
                                        color_k = forward_sample_texture(textures_t, w_clip, texture_res, k, texture_type, fn, bn)
                                        soft_color(k) = soft_color(k) + coef * color_k
                                    end do
                                end if
                            end if
                        end if
                    end if
                end do

                soft_colors_t((bn - 1) * (4 * image_size * image_size) + &
                        (4 - 1) * (image_size * image_size) + pn) = 1.0 - soft_color_alpha(num_faces)
                do k = 1, 3
                    soft_colors_t((bn - 1) * (4 * image_size * image_size) + &
                        (k - 1) * (image_size * image_size) + pn) = soft_color(k) / softmax_sum
                end do
            end do
            !$OMP END PARALLEL DO
        end do
    end subroutine tapenade_render_main
end Module tapenade_render
