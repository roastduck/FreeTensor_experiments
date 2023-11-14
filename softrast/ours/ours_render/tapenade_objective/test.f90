Module tapenade_render
    implicit none

    ! Macro Definitions
    integer, parameter :: batch_size = 1
    integer, parameter :: num_faces = 5856
    integer, parameter :: texture_size = 25
    integer, parameter :: texture_res = 5
    integer, parameter :: image_size = 256
    real*8, parameter :: near = 1.0
    real*8, parameter :: far = 100.0
    real*8, parameter :: eps = 1e-3
    real*8, parameter :: sigma_val = 1e-5
    real*8, parameter :: gamma_val = 1e-4
    real*8, parameter :: dist_eps = 9.21024036697585
    real*8, parameter :: threshold = dist_eps * sigma_val
    logical, parameter :: double_side = .false.
    integer, parameter :: texture_type = 0

contains
    subroutine tapenade_render_main(soft_colors_t) BIND(c)
        integer, intent(inout) :: soft_colors_t(image_size * image_size * 4 * batch_size)
        do bn = 1, batch_size
            do k = 1, 4
                do pn = 1, image_size * image_size
                    soft_colors_t((bn - 1) * (4 * image_size * image_size) + &
                        (k - 1) * (image_size * image_size) + pn) = (bn - 1) * (4 * image_size * image_size) + &
                        (k - 1) * (image_size * image_size) + pn - 1;
                end do
            end do
        end do
    end subroutine tapenade_render_main
end Module tapenade_render
