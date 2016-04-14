/*
 * NOTICE
 *
 * This work was produced at the Lawrence Livermore National Laboratory (LLNL)
 * under contract no. DE-AC-52-07NA27344 (Contract 44) between the U.S.
 * Department of Energy (DOE) and Lawrence Livermore National Security, LLC
 * (LLNS) for the operation of LLNL. The rights of the Federal Government are
 * reserved under Contract 44.
 *
 * DISCLAIMER
 *
 * This work was prepared as an account of work sponsored by an agency of the
 * United States Government. Neither the United States Government nor Lawrence
 * Livermore National Security, LLC nor any of their employees, makes any
 * warranty, express or implied, or assumes any liability or responsibility
 * for the accuracy, completeness, or usefulness of any information, apparatus,
 * product, or process disclosed, or represents that its use would not infringe
 * privately-owned rights. Reference herein to any specific commercial products,
 * process, or service by trade name, trademark, manufacturer or otherwise does
 * not necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * NOTIFICATION OF COMMERCIAL USE
 *
 * Commercialization of this product is prohibited without notifying the
 * Department of Energy (DOE) or Lawrence Livermore National Security.
 */

#include<Kripke.h>
#include<Kripke/Input_Variables.h>
#include<Kripke/Grid.h>
#include<Kripke/Test/TestKernels.h>
#include<stdio.h>
#include<string.h>
#include<mpi.h>
#include<algorithm>
#include<string>
#include<sstream>

#ifdef KRIPKE_USE_OPENMP
#include<omp.h>
#endif

#ifdef KRIPKE_USE_TCMALLOC
#include<gperftools/malloc_extension.h>
#endif

#ifdef KRIPKE_USE_PERFTOOLS
#include<gperftools/profiler.h>
#endif

#ifdef __bgq__
#include </bgsys/drivers/ppcfloor/spi/include/kernel/location.h>
#include </bgsys/drivers/ppcfloor/spi/include/kernel/memory.h>
#endif


void usage(void){
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid == 0){
    // Get a new object with defaulted values
    Input_Variables def;
    
    // Display command line
    printf("Usage:  [srun ...] kripke [options...]\n\n");
    
    // Display each option
    printf("Problem Size Options:\n");
    printf("---------------------\n");
    
    printf("  --groups <ngroups>     Number of energy groups\n");
    printf("                         Default:  --groups %d\n\n", def.num_groups);
    
    printf("  --legendre <lorder>    Scattering Legendre Expansion Order (0, 1, ...)\n");
    printf("                         Default:  --legendre %d\n\n", def.legendre_order);
    
    printf("  --quad [<ndirs>|<polar>:<azim>]\n");
    printf("                         Define the quadrature set to use\n");
    printf("                         Either a fake S2 with <ndirs> points,\n");
    printf("                         OR Gauss-Legendre with <polar> by <azim> points\n");
    printf("                         Default:  --quad %d\n\n", def.num_directions);
    
    
    
    printf("  --zones <x,y,z>        Number of zones in x,y,z\n");
    printf("                         Default:  --zones %d,%d,%d\n\n", def.nx, def.ny, def.nz);
    
    
    printf("\n");
    printf("Physics Parameters:\n");
    printf("-------------------\n");
    printf("  --sigt <st0,st1,st2>   Total material cross-sections\n");
    printf("                         Default:   --sigt %lf,%lf,%lf\n\n", def.sigt[0], def.sigt[1], def.sigt[2]);

    printf("  --sigs <ss0,ss1,ss2>   Scattering material cross-sections\n");
    printf("                         Default:   --sigs %lf,%lf,%lf\n\n", def.sigs[0], def.sigs[1], def.sigs[2]);


    printf("\n");
    printf("On-Node Options:\n");
    printf("----------------\n");
    printf("  --nest <NEST>          Loop nesting order (and data layout)\n");
    printf("                         Available: DGZ,DZG,GDZ,GZD,ZDG,ZGD\n");
    printf("                         Default:   --nest %s\n\n", nestingString(def.nesting).c_str());
    
    printf("\n");
    printf("Parallel Decomposition Options:\n");
    printf("-------------------------------\n");
    printf("  --layout <lout>        Layout of spatial subdomains over mpi ranks\n");
    printf("                         0: Blocked: local zone sets are adjacent\n");
    printf("                         1: Scattered: adjacent zone sets are distributed\n");
    printf("                         Default: --layout %d\n\n", def.layout_pattern);
    
    
    printf("  --procs <npx,npy,npz>  Number of MPI ranks in each spatial dimension\n");
    printf("                         Default:  --procs %d,%d,%d\n\n", def.npx, def.npy, def.npz);
    
    printf("  --dset <ds>            Number of direction-sets\n");
    printf("                         Must be a factor of 8, and divide evenly the number\n");
    printf("                         of quadrature points\n");
    printf("                         Default:  --dset %d\n\n", def.num_dirsets);
    
    printf("  --gset <gs>            Number of energy group-sets\n");
    printf("                         Must divide evenly the number energy groups\n");
    printf("                         Default:  --gset %d\n\n", def.num_groupsets);
    
    printf("  --zset <zx>,<zy>,<zz>  Number of zone-sets in x,y, and z\n");
    printf("                         Default:  --zset %d,%d,%d\n\n", def.num_zonesets_dim[0], def.num_zonesets_dim[1], def.num_zonesets_dim[2]);
    
    printf("\n");
    printf("Solver Options:\n");
    printf("---------------\n");
    
    printf("  --niter <NITER>        Number of solver iterations to run\n");
    printf("                         Default:  --niter %d\n\n", def.niter);
    
    printf("  --pmethod <method>     Parallel solver method\n");
    printf("                         sweep: Full up-wind sweep (wavefront algorithm)\n");
    printf("                         bj: Block Jacobi\n");
    printf("                         Default: --pmethod sweep\n\n");
    
    printf("\n");
    printf("Output and Testing Options:\n");
    printf("---------------------------\n");
    
#ifdef KRIPKE_USE_PAPI
    printf("  --papi <PAPI_X_X,...>  Track PAPI hardware counters for each timer\n\n");
#endif
#ifdef KRIPKE_USE_SILO
    printf("  --silo <BASENAME>      Create SILO output files\n\n");
#endif
    printf("  --test                 Run Kernel Test instead of solver\n\n");
    printf("\n");
  }
  MPI_Finalize();
  exit(1);
}

struct CmdLine {
  CmdLine(int argc, char **argv) :
    size(argc-1),
    cur(0),
    args()
  {
    for(int i = 0;i < size;++ i){
      args.push_back(argv[i+1]);
    }
  }

  std::string pop(void){
    if(atEnd())
      usage();
    return args[cur++];
  }

  bool atEnd(void){
    return(cur >= size);
  }

  int size;
  int cur;
  std::vector<std::string> args;
};

std::vector<std::string> split(std::string const &str, char delim){
  std::vector<std::string> elem;
  std::stringstream ss(str);
  std::string e;
  while(std::getline(ss, e, delim)){
    elem.push_back(e);
  }
  return elem;
}


namespace {
  template<typename T>
  std::string toString(T const &val){
    std::stringstream ss;
    ss << val;
    return ss.str();
  }
}

int main(int argc, char **argv) {
  /*
   * Initialize MPI
   */
  MPI_Init(&argc, &argv);
  int myid;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  int num_tasks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

  if (myid == 0) {
    /* Print out a banner message along with a version number. */
    printf("\n");
    printf("----------------------------------------------------------------------\n");
    printf("------------------------ KRIPKE VERSION 1.1 --------------------------\n");
    printf("----------------------------------------------------------------------\n");
    printf("This work was produced at the Lawrence Livermore National Laboratory\n");
    printf("(LLNL) under contract no. DE-AC-52-07NA27344 (Contract 44) between the\n");
    printf("U.S. Department of Energy (DOE) and Lawrence Livermore National\n");
    printf("Security, LLC (LLNS) for the operation of LLNL. The rights of the\n");
    printf("Federal Government are reserved under Contract 44.\n");
    printf("\n");
    printf("Main Contact: Adam J. Kunen <kunen1@llnl.gov>\n");
    printf("----------------------------------------------------------------------\n");
   
   
    /* Print out some information about how OpenMP threads are being mapped
     * to CPU cores.
     */
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
#ifdef __bgq__
      int core = Kernel_ProcessorCoreID();
#else
      int core = sched_getcpu();
#endif
      printf("Rank: %d Thread %d: Core %d\n", myid, tid, core);
    }
#endif
  }

  /*
   * Default input parameters
   */
  Input_Variables vars;
  std::vector<std::string> papi_names;
  bool test = false;
  
  /*
   * Parse command line
   */
  CmdLine cmd(argc, argv);
  while(!cmd.atEnd()){
    std::string opt = cmd.pop();
    if(opt == "-h" || opt == "--help"){usage();}
    else if(opt == "--name"){vars.run_name = cmd.pop();}
    else if(opt == "--dset"){
      vars.num_dirsets = std::atoi(cmd.pop().c_str());      
    }
    else if(opt == "--gset"){
      vars.num_groupsets = std::atoi(cmd.pop().c_str());      
    }
    else if(opt == "--zset"){
      std::vector<std::string> nz = split(cmd.pop(), ',');
      if(nz.size() != 3) usage();
      vars.num_zonesets_dim[0] = std::atoi(nz[0].c_str());
      vars.num_zonesets_dim[1] = std::atoi(nz[1].c_str());
      vars.num_zonesets_dim[2] = std::atoi(nz[2].c_str());      
    }
    else if(opt == "--layout"){
      vars.layout_pattern = std::atoi(cmd.pop().c_str());      
    }
    else if(opt == "--zones"){
      std::vector<std::string> nz = split(cmd.pop(), ',');
      if(nz.size() != 3) usage();
      vars.nx = std::atoi(nz[0].c_str());
      vars.ny = std::atoi(nz[1].c_str());
      vars.nz = std::atoi(nz[2].c_str());
    }
    else if(opt == "--procs"){
      std::vector<std::string> np = split(cmd.pop(), ',');
      if(np.size() != 3) usage();
      vars.npx = std::atoi(np[0].c_str());
      vars.npy = std::atoi(np[1].c_str());
      vars.npz = std::atoi(np[2].c_str());
    }
    else if(opt == "--pmethod"){
      std::string method = cmd.pop();
      if(!strcasecmp(method.c_str(), "sweep")){
        vars.parallel_method = PMETHOD_SWEEP;
      }
      else if(!strcasecmp(method.c_str(), "bj")){
        vars.parallel_method = PMETHOD_BJ;
      }
      else{
        usage();
      }
    }
    else if(opt == "--groups"){
      vars.num_groups = std::atoi(cmd.pop().c_str());      
    }
    else if(opt == "--quad"){
      std::vector<std::string> p = split(cmd.pop(), ':');
      if(p.size() == 1){
        vars.num_directions = std::atoi(p[0].c_str());
        vars.quad_num_polar = 0;
        vars.quad_num_azimuthal = 0;
      }
      else if(p.size() == 2){
        vars.quad_num_polar = std::atoi(p[0].c_str());
        vars.quad_num_azimuthal = std::atoi(p[1].c_str());
        vars.num_directions = vars.quad_num_polar * vars.quad_num_azimuthal;
      }
      else{
        usage();
      }
    }
    else if(opt == "--legendre"){
      vars.legendre_order = std::atoi(cmd.pop().c_str());
    }
    else if(opt == "--sigs"){
      std::vector<std::string> values = split(cmd.pop(), ',');
      if(values.size()!=3)usage();
      for(int mat = 0;mat < 3;++ mat){
        vars.sigs[mat] = std::atof(values[mat].c_str());
      }
    }
    else if(opt == "--sigt"){
      std::vector<std::string> values = split(cmd.pop(), ',');
      if(values.size()!=3)usage();
      for(int mat = 0;mat < 3;++ mat){
        vars.sigt[mat] = std::atof(values[mat].c_str());
      }
    }
    else if(opt == "--niter"){
      vars.niter = std::atoi(cmd.pop().c_str());
    }
    else if(opt == "--nest"){
      vars.nesting = nestingFromString(cmd.pop());     
    }
#ifdef KRIPKE_USE_SILO
    else if(opt == "--silo"){
      vars.silo_basename = cmd.pop();
    }
#endif
    else if(opt == "--test"){
      test = true;
    }
#ifdef KRIPKE_USE_PAPI
    else if(opt == "--papi"){
      papi_names = split(cmd.pop(), ',');
    }
#endif
    else{
      printf("Unknwon options %s\n", opt.c_str());
      usage();
    }
  }
  
  // Check that the input arguments are valid
  if(vars.checkValues()){
    exit(1);
  }

  /*
   * Display Options
   */
  if (myid == 0) {
    printf("Number of MPI tasks:   %d\n", num_tasks);
#ifdef KRIPKE_USE_OPENMP
    int num_threads=1;
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
      if(omp_get_thread_num() == 0){
          printf("OpenMP threads/task:   %d\n", num_threads);
          printf("OpenMP total threads:  %d\n", num_threads*num_tasks);
        }
    }
#endif

#ifdef KRIPKE_USE_PAPI
    printf("PAPI Counters:         ");
    if(papi_names.size() > 0){
      for(int i = 0;i < papi_names.size();++ i){
        printf("%s ", papi_names[i].c_str());
      }
    }
    else{
      printf("<none>");
    }
    printf("\n");
#endif
    printf("Processors:            %d x %d x %d\n", vars.npx, vars.npy, vars.npz);
    printf("Zones:                 %d x %d x %d\n", vars.nx, vars.ny, vars.nz);
    printf("Legendre Order:        %d\n", vars.legendre_order);
    printf("Total X-Sec:           sigt=[%lf, %lf, %lf]\n", vars.sigt[0], vars.sigt[1], vars.sigt[2]);
    printf("Scattering X-Sec:      sigs=[%lf, %lf, %lf]\n", vars.sigs[0], vars.sigs[1], vars.sigs[2]);
    printf("Quadrature Set:        ");
    if(vars.quad_num_polar == 0){
      printf("Dummy S2 with %d points\n", vars.num_directions);
    }
    else {
      printf("Gauss-Legendre, %d polar, %d azimuthal (%d points)\n", vars.quad_num_polar, vars.quad_num_azimuthal, vars.num_directions);
    }
    printf("Parallel method:       ");
    if(vars.parallel_method == PMETHOD_SWEEP){
      printf("Sweep\n");
    }
    else if(vars.parallel_method == PMETHOD_BJ){
      printf("Block Jacobi\n");
    }
    printf("Loop Nesting Order     %s\n", nestingString(vars.nesting).c_str());        
    printf("Number iterations:     %d\n", vars.niter);
    
    printf("GroupSet/Groups:       %d sets, %d groups/set\n", vars.num_groupsets, vars.num_groups/vars.num_groupsets);
    printf("DirSets/Directions:    %d sets, %d directions/set\n", vars.num_dirsets, vars.num_directions/vars.num_dirsets);

    printf("Zone Sets:             %d,%d,%d\n", vars.num_zonesets_dim[0], vars.num_zonesets_dim[1], vars.num_zonesets_dim[2]);

    
  }

#ifdef KRIPKE_USE_PERFTOOLS
  ProfilerStart("kripke.prof");
#endif  

  if(test){
    // Invoke Kernel testing
    testKernels(vars);
  }
  else{
    // Allocate problem 
    Grid_Data *grid_data = new Grid_Data(&vars);

    grid_data->timing.setPapiEvents(papi_names);

    // Run the solver
    SweepSolver(grid_data, vars.parallel_method == PMETHOD_BJ);

#ifdef KRIPKE_USE_SILO
    // Output silo data
    if(vars.silo_basename != ""){
      grid_data->writeSilo(vars.silo_basename);
    }
#endif

    // Print Timing Info
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(myid == 0){
      grid_data->timing.print();
      printf("\n\n");
    }

    // Cleanup 
    delete grid_data;
  }
  
#ifdef KRIPKE_USE_PERFTOOLS
  ProfilerStop();
#endif  

  // Gather post-point memory info
  double heap_mb = -1.0;
  double hwm_mb = -1.0;
#ifdef KRIPKE_USE_TCMALLOC
  // If we are using tcmalloc, we need to use it's interface
  MallocExtension *mext = MallocExtension::instance();
  size_t bytes;

  mext->GetNumericProperty("generic.current_allocated_bytes", &bytes);
  heap_mb = ((double)bytes)/1024.0/1024.0;

  mext->GetNumericProperty("generic.heap_size", &bytes);
  hwm_mb = ((double)bytes)/1024.0/1024.0;
#else
#ifdef __bgq__
  // use BG/Q specific calls (if NOT using tcmalloc)
  uint64_t bytes;

  int rc = Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP, &bytes);
  heap_mb = ((double)bytes)/1024.0/1024.0;

  rc = Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPMAX, &bytes);
  hwm_mb = ((double)bytes)/1024.0/1024.0;
#endif
#endif
  // Print memory info
  if(myid == 0 && heap_mb >= 0.0){
    printf("Bytes allocated: %lf MB\n", heap_mb);
    printf("Heap Size      : %lf MB\n", hwm_mb);

  }
  
  // Cleanup and exit
  MPI_Finalize();

  return (0);
}
