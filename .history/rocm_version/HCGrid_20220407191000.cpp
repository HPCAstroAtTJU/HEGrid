// --------------------------------------------------------------------
//
// title                  :HCGrid.cpp
// description            :Grid data points to map
// author                 :
//
// --------------------------------------------------------------------

#include "HCGrid.h"

int main(int argc, char **argv){
    // Get FITS files from command
    char *path = NULL, *ifile = NULL, *tfile = NULL, *ofile = NULL, *sfile = NULL, *num = NULL,\
    *beam = NULL, *order = NULL, *bDim = NULL, *Rg= NULL, *Sp = NULL, *factor = NULL;
    char pcl;
    int option_index = 0;
    static const struct option long_options[] = {
        {"helparg", no_argument, NULL, 'h'},
        {"fits_path", required_argument, NULL, 'p'},            // absolute path of FITS file
        {"input_file", required_argument, NULL, 'i'},           // name of unsorted input FITS file (it will call sort function)
        {"target_file", required_argument, NULL, 't'},          // name of target FITS file
        {"output_file", required_argument, NULL, 'o'},          // name of output FITS file
        {"sorted_file", required_argument, NULL, 's'},          // name of sorted input FITS file (it won't call sort function)
        {"file_id", required_argument, NULL, 'n'},              // ID of FITS file
        {"beam_size", required_argument, NULL, 'b'},            // beam size of FITS file
        {"order_arg", required_argument, NULL, 'd'},            // sort parameter
        {"block_num", required_argument, NULL, 'a'},            // the number of thread in each block
        {"register_num", required_argument, NULL, 'r'},        // the number of register in SM
        {"sp_num", required_argument, NULL, 'm'},              // the number of SP in each SM
        {"coarsening_factor", required_argument, NULL, 'f'},    // the value of coarsening factor
        {0, 0, 0, 0}
    };

    while((pcl = getopt_long_only (argc, argv, "hp:i:t:o:s:n:b:d:a:r:m:f:", long_options, \
                    &option_index)) != EOF){
        switch(pcl){
            case 'h':
                fprintf(stderr, "useage: ./HCGrid --fits_path <absolute path> --input_file <input file> --target_file <target file> "
                "--sorted_file <sorted file> --output_file <output file> --file_id <number> --beam_size <beam> --order_arg <order> --block_num <num>  --coarsening_factor<factor> \n");
                return 1;
            case 'p':
                path = optarg;
                break;
            case 'i':
                ifile = optarg;
                break;
            case 't':
                tfile = optarg;
                break;
            case 'o':
                ofile = optarg;
                break;
            case 's':
                sfile = optarg;
                break;
            case 'n':
                num = optarg;
                break;
            case 'b':
                beam = optarg;
                break;
            case 'd':
                order = optarg;
                break;
            case 'a':
                bDim = optarg;
                break;
            case 'r':
                Rg = optarg;
                break;
            case 'm':
                Sp = optarg;
                break;
            case 'f':
                factor = optarg;
                break;
            case '?':
                fprintf (stderr, "Unknown option `-%c'.\n", (char)optopt);
                break;
            default:
                return 1;
        }
    }

    char infile[180] = "", tarfile[180] = "", outfile[180] = "!", sortfile[180] = "!";
    strcat(infile, path);
    strcat(infile, ifile);
    strcat(infile, num);
    // strcat(infile, ".fits");
    strcat(infile, ".hdf5");
    strcat(tarfile, path);
    strcat(tarfile, tfile);
    // strcat(tarfile, num);
    strcat(tarfile, ".fits");
    strcat(outfile, path);
    strcat(outfile, ofile);
    strcat(outfile, num);
    strcat(outfile, ".fits");
    if (sfile) {
        strcat(sortfile, path);
        strcat(sortfile, sfile);
        strcat(sortfile, num);
        strcat(sortfile, ".fits");
    }
    // printf("order: %s, num: %s, ", order, num);

    double hostTime1 = cpuSecond();

    // Initialize healpix
    _Healpix_init(1, RING);

    // Set kernel
    uint32_t kernel_type = GAUSS1D;
    double kernelsize_fwhm = 180. / 3600.;
    if (beam) {
        kernelsize_fwhm = atoi(beam) / 3600.;
    }
    double kernelsize_sigma = kernelsize_fwhm / sqrt(8*log(2));
    double *kernel_params;
    kernel_params = RALLOC(double, 3);
    kernel_params[0] = kernelsize_sigma;
    double sphere_radius = 5 * kernelsize_sigma;
    double hpx_max_resolution = kernelsize_sigma / 2.;
    _prepare_grid_kernel(kernel_type, kernel_params, sphere_radius, hpx_max_resolution);

    // Get Register num and SP num
    int Register, SP, T_max, T_max_h, BlockDim_x;

    // Gridding process
    h_GMaps.factor = 1;
    if (factor) {
        h_GMaps.factor = atoi(factor);
    }
    // printf("h_GMaps.factor=%d, ", h_GMaps.factor);
    if (sfile) {
        if (bDim){
            solve_gridding(infile, tarfile, outfile, sortfile, atoi(order), atoi(bDim), argc, argv );
        }else if ( Rg && Sp){
                Register = atoi(Rg) * 1024;
                SP = atoi(Sp);
                T_max = Register / 184;
                T_max_h = T_max / 2;
                if ((SP >= 32) && (SP < T_max_h))
                    BlockDim_x = SP;
                else
                    BlockDim_x = T_max - T_max % 32;
            // printf("blockDim_x=%d\n", BlockDim_x);
            solve_gridding(infile, tarfile, outfile, sortfile, atoi(order), BlockDim_x, argc, argv);
        } else
            solve_gridding(infile, tarfile, outfile, sortfile, atoi(order), 64, argc, argv);
    } else {
        if (bDim){
            solve_gridding(infile, tarfile, outfile, NULL, atoi(order), atoi(bDim), argc, argv);
        } else if ( Rg && Sp){
                Register = atoi(Rg) * 1024;
                SP = atoi(Sp);
                T_max = Register / 184;
                T_max_h = T_max / 2;
                if ((SP >= 32) && (SP < T_max_h))
                    BlockDim_x = SP;
                else
                    BlockDim_x = T_max - T_max % 32;
            solve_gridding(infile, tarfile, outfile, NULL, atoi(order), BlockDim_x, argc, argv);
        }
        else
            solve_gridding(infile, tarfile, outfile, NULL, atoi(order), 64, argc, argv);
    }

    double hostTime2 = cpuSecond();
    float elaspTime = (hostTime2 - hostTime1) * 1000;
    // printf("Running Time = %f\n", elaspTime);
    // printf("%f\n", elaspTime);
    // printf("**************************************************\n");  
    return 0;
}
