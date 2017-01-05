#include <string>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "supervoxel_registration.h"
#include "supervoxel_util.hpp"

struct options {
	float vr;
	float sr;
	float colorWeight;
	float spatialWeight;
	float normalWeight;
	int test;
	bool showScans;
	bool approx;
	std::string dataDir;
	std::string transformFile;
	bool debug;
} programOptions;

int initOptions(int argc, char* argv[]) {

	namespace po = boost::program_options;

	programOptions.colorWeight = 0.0f;
	//programOptions.spatialWeight = 0.4f;
	programOptions.spatialWeight = 1.0f;
	programOptions.normalWeight = 1.0f;
	programOptions.vr = 1.0f;
	programOptions.sr = 5.0f;
	programOptions.test = 0; //v-5: 324,180_185: 607
	programOptions.showScans = false;
	programOptions.approx = false;
	programOptions.dataDir = "../data/";
	programOptions.transformFile = "trans_base";
	programOptions.debug = true;

	po::options_description desc ("Allowed Options");

	desc.add_options()
					("help,h", "Usage <Scan 1 Path> <Scan 2 Path> <Transform File>")
					("voxel_res,v", po::value<float>(&programOptions.vr), "voxel resolution")
					("seed_res,s", po::value<float>(&programOptions.sr), "seed resolution")
					("color_weight,c", po::value<float>(&programOptions.colorWeight), "color weight")
					("spatial_weight,z", po::value<float>(&programOptions.spatialWeight), "spatial weight")
					("normal_weight,n", po::value<float>(&programOptions.normalWeight), "normal weight")
					("test,t", po::value<int>(&programOptions.test), "test")
					("show_scan,y", po::value<bool>(&programOptions.showScans), "Show scans")
					("data_dir", po::value<std::string>(&programOptions.dataDir), "Data Directory for the transformation")
					("transform_file,f", po::value<std::string>(&programOptions.transformFile), "Transform file name")
					("approximate,p", po::value<bool>(&programOptions.approx), "Approximate angles");

	po::variables_map vm;

	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help")) {
		std::cout << "Supervoxel MI based Scan Alignment" << std::endl << desc << std::endl;
		return 1;
	} else {

		if (programOptions.debug) {
			std::cout << "vr: " << programOptions.vr << std::endl;
			std::cout << "sr: " << programOptions.sr << std::endl;
			std::cout << "colorWeight: " << programOptions.colorWeight << std::endl;
			std::cout << "spatialWeight: " << programOptions.spatialWeight << std::endl;
			std::cout << "normalWeight: " << programOptions.normalWeight << std::endl;
		}

		return 0;
	}

}

int
main (int argc, char *argv[]) {

	if (initOptions(argc, argv))
		return 1;

	if (argc < 2) {
		std::cerr << "One or more scan files/transform missing" << std::endl;
		return 1;
	}

	int s1, s2;

	s1 = atoi(argv[1]);
	s2 = atoi(argv[2]);

	std::string transformFile = programOptions.transformFile;
	const std::string dataDir = programOptions.dataDir;//"../data/";

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scanA = boost::shared_ptr <pcl::PointCloud<pcl::PointXYZRGBA> > (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scanB = boost::shared_ptr <pcl::PointCloud<pcl::PointXYZRGBA> > (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr temp = boost::shared_ptr <pcl::PointCloud<pcl::PointXYZRGBA> > (new pcl::PointCloud<pcl::PointXYZRGBA> ());

	std::stringstream ss;

	ss << dataDir << boost::format("scans_pcd/scan_%04d.pcd")%s1;

	if (pcl::io::loadPCDFile<pcl::PointXYZRGBA> (ss.str(), *scanA)) {
		std::cout << "Error loading cloud file: " << ss.str() << std::endl;
		return (1);
	}

	// clear string
	ss.str(std::string());
	ss << dataDir << boost::format("scans_pcd/scan_%04d.pcd")%s2;

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

	svr::SupervoxelRegistration supervoxelRegistration (programOptions.vr, programOptions.sr);
	supervoxelRegistration.setDebug(programOptions.debug);
	supervoxelRegistration.setDebugScanString((boost::format("%d_%d")%s1%s2).str());
	supervoxelRegistration.setApproximate(programOptions.approx);

	if (programOptions.test != 0 || programOptions.showScans) {

		pcl::transformPointCloud (*temp, *scanB, transform.inverse());
		temp->clear();
		supervoxelRegistration.setScans(scanA, scanB);
		std::cout << "Testing..." << std::endl;

		if (programOptions.showScans && programOptions.test == 0) {
			supervoxelRegistration.showPointClouds("Supervoxel Based MI Viewer: " + transformFile);
			return 0;
		} else {
			supervoxelRegistration.showTestSuperVoxel(programOptions.test);
		}

	} else {

		scanB = temp;
		supervoxelRegistration.setScans(scanA, scanB);

		clock_t start = svr_util::getClock();

		// begin registration
		Eigen::Affine3d r;
		Eigen::Affine3d result;
		Eigen::Affine3d initialT = transform.inverse();

		supervoxelRegistration.alignScans(r, initialT);

		// Save transformation in a file

//		result = transform.matrix() * result.inverse();

		result = r.inverse();
//		Eigen::Matrix4d result = trans_new.inverse().matrix();
		std::cout << "Resultant Transformation: " << std::endl << result.matrix();
		std::cout << std::endl;


		// clear string
		ss.str(std::string());

		if (!programOptions.approx)
			ss << dataDir << "/trans_absolute";
		else
			ss << dataDir << "/trans_approx";

		if(!(boost::filesystem::exists(ss.str()))){
			std::cout<<"Directory doesn't Exist"<<std::endl;

			if (boost::filesystem::create_directory(ss.str()))
				std::cout << "....Successfully Created !" << std::endl;
		}

		ss << "/trans_result_" << s1 << "_" << s2;
		std::string transResultFile = ss.str();
		std::ofstream fout(transResultFile.c_str());
		fout << result.matrix();
		fout.close();

		clock_t end = svr_util::getClock();

		if (programOptions.debug)
			std::cout << "Total time: " << svr_util::getClockTime(start, end) << std::endl;

	}

}











