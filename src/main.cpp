#include <string>
#include "supervoxel_registration.h"
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

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
	programOptions.test = 0; // 324 // 607
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
		std::cout << "Supervoxel MI based Scan Alignment" << std::endl << desc << std::endl;
		return 1;
	} else {

		std::cout << "vr: " << programOptions.vr << std::endl;
		std::cout << "sr: " << programOptions.sr << std::endl;
		std::cout << "colorWeight: " << programOptions.colorWeight << std::endl;
		std::cout << "spatialWeight: " << programOptions.spatialWeight << std::endl;
		std::cout << "normalWeight: " << programOptions.normalWeight << std::endl;

		return 0;
	}

}

int
main (int argc, char *argv[]) {

	if (initOptions(argc, argv))
		return 1;

	if (argc < 3) {
		std::cerr << "One or more scan files/transform missing" << std::endl;
		return 1;
	}

	int s1, s2;
	std::string transformFile;

	s1 = atoi(argv[1]);
	s2 = atoi(argv[2]);

	if (argc > 3)
		transformFile = argv[3];

	const std::string dataDir = "../data/";

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scanA = boost::shared_ptr <pcl::PointCloud<pcl::PointXYZRGBA> > (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scanB = boost::shared_ptr <pcl::PointCloud<pcl::PointXYZRGBA> > (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr temp = boost::shared_ptr <pcl::PointCloud<pcl::PointXYZRGBA> > (new pcl::PointCloud<pcl::PointXYZRGBA> ());

	std::cout << "Loading PointClouds..." << std::endl;

	std::stringstream ss;

	ss << dataDir << boost::format("scan_%04d.pcd")%s1;

	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (ss.str(), *scanA)) {
		std::cout << "Error loading cloud file: " << ss.str() << std::endl;
		return (1);
	}

	// clear string
	ss.str(std::string());
	ss << dataDir << boost::format("scan_%04d.pcd")%s2;

	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (ss.str(), *temp)) {
		std::cout << "Error loading cloud file: " << ss.str() << std::endl;
		return (1);
	}

	Eigen::Affine3d transform = Eigen::Affine3d::Identity();

	if (transformFile.size() != 0) {

		ss.str(std::string());
		ss << dataDir << transformFile;

		transformFile = ss.str();

		std::ifstream in(transformFile.c_str());
		if (!in) {
			std::stringstream err;
			err << "Error loading transformation " << transformFile.c_str() << std::endl;
			std::cerr << err.str();
			return 1;
		}

		std::string line;

		for (int i = 0; i < 4; i++) {
			std::getline(in,line);

			std::istringstream sin(line);
			for (int j = 0; j < 4; j++) {
				sin >> transform (i,j);
			}
		}

	}

	// transform
	pcl::transformPointCloud (*temp, *scanB, transform.inverse());
	temp->clear();

	svr::SupervoxelRegistration supervoxelRegistration (programOptions.vr, programOptions.sr);
	supervoxelRegistration.setScans(scanA, scanB);

	if (programOptions.test != 0 || programOptions.showScans) {

		std::cout << "Testing..." << std::endl;

		if (programOptions.showScans && programOptions.test == 0) {
			supervoxelRegistration.showPointClouds("Supervoxel Based MI Viewer: " + transformFile);
			return 0;
		} else {
			supervoxelRegistration.showTestSuperVoxel(programOptions.test);
		}

	} else {

		std::cout << "Alignment Scan" << std::endl;

		// begin registration
		supervoxelRegistration.alignScans();
	}

}











