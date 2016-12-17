#include <string>

struct options {

	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;
	bool showScans;

} programOptions;

int initOptions(int argc, char* argv[]) {

	namespace po = boost::program_options;

	programOptions.colorWeight = 0.0f;
	programOptions.spatialWeight = 0.4f;
	programOptions.normalWeight = 1.0f;
	programOptions.vr = 1.0f;
	programOptions.sr = 5.0f;
	programOptions.test = 0; // 324
	programOptions.showScans = false;

	po::options_description desc ("Allowed Options");

	desc.add_options()
													("help,h", "Usage <Scan 1 Path> <Scan 2 Path> <Transform File>")
													("voxel_res,v", po::value<float>(&programOptions.vr), "voxel resolution")
													("seed_res,s", po::value<float>(&programOptions.sr), "seed resolution")
													("color_weight,c", po::value<float>(&programOptions.colorWeight), "color weight")
													("spatial_weight,z", po::value<float>(&programOptions.spatialWeight), "spatial weight")
													("normal_weight,n", po::value<float>(&programOptions.normalWeight), "normal weight")
													("test,t", po::value<int>(&programOptions.test), "test")
													("show_scan,y", po::value<bool>(&programOptions.showScans), "Show scans");

	po::variables_map vm;

	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		cout << "Supervoxel MI based SCan Alignment" << endl << desc << endl;
		return 1;
	} else {

		cout << "vr: " << programOptions.vr << endl;
		cout << "sr: " << programOptions.sr << endl;
		cout << "colorWeight: " << programOptions.colorWeight << endl;
		cout << "spatialWeight: " << programOptions.spatialWeight << endl;
		cout << "normalWeight: " << programOptions.normalWeight << endl;

		return 0;
	}

}
