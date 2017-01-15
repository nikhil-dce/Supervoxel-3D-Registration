#include <string>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_plotter.h>

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

	bool plotAverage;
	int plotResultsStep;

	bool plotOptimizationIterations;
	bool plotX;
	bool plotY;
	bool plotZ;
	bool plotRoll;
	bool plotPitch;
	bool plotYaw;

	float plotStartTrans;
	float plotEndTrans;
	float plotStepTrans;

	float plotStartAngle;
	float plotEndAngle;
	float plotStepAngle;

} programOptions;

int initOptions(int argc, char* argv[]) {

	namespace po = boost::program_options;

	programOptions.colorWeight = 0.0f;
	programOptions.spatialWeight = 1.0f;
	programOptions.normalWeight = 1.0f;
	programOptions.vr = 1.0f;
	programOptions.sr = 10.0f;
	programOptions.test = 0; //v-5: 324,180_185: 607
	programOptions.showScans = false;
	programOptions.approx = false;
	programOptions.dataDir = "../data/";
	programOptions.transformFile = "trans_base";
	programOptions.debug = true;

	programOptions.plotAverage = false;
	programOptions.plotResultsStep = 0;

	programOptions.plotX = false;
	programOptions.plotY = false;
	programOptions.plotZ = false;
	programOptions.plotRoll = false;
	programOptions.plotPitch = false;
	programOptions.plotYaw = false;
	programOptions.plotOptimizationIterations = true;

	programOptions.plotStartTrans = 0;
	programOptions.plotEndTrans = 0;
	programOptions.plotStepTrans = 1;

	programOptions.plotStartAngle = 0;
	programOptions.plotEndAngle = 0;
	programOptions.plotStepAngle = 1;


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

					("plotAverageGraph", po::value<bool>(&programOptions.plotAverage), "Plot Average Results for 5,10,15 and 20 scans apart")
					("plotResultsStep", po::value<int>(&programOptions.plotResultsStep), "Plot Results for 5,10,15 or 20 scans apart")

					("plotX", po::value<bool>(&programOptions.plotX), "PlotX Score Data")
					("plotY", po::value<bool>(&programOptions.plotY), "PlotY Score Data")
					("plotZ", po::value<bool>(&programOptions.plotZ), "PlotZ Score Data")
					("plotRoll", po::value<bool>(&programOptions.plotRoll), "PlotRoll Score Data")
					("plotPitch", po::value<bool>(&programOptions.plotPitch), "PlotPitch Score Data")
					("plotYaw", po::value<bool>(&programOptions.plotYaw), "PlotYaw Score Data")

					("plotStartTrans", po::value<float>(&programOptions.plotStartTrans), "PlotTrans Start")
					("plotEndTrans", po::value<float>(&programOptions.plotEndTrans), "PlotTrans End")
					("plotStepTrans", po::value<float>(&programOptions.plotStepTrans), "PlotTrans Step Size")

					("plotStartAngle", po::value<float>(&programOptions.plotStartAngle), "PlotAngle Start")
					("plotEndAngle", po::value<float>(&programOptions.plotEndAngle), "PlotAngle End")
					("plotStepAngle", po::value<float>(&programOptions.plotStepAngle), "PlotAngle Step Size")

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

void loadTransform(std::string filename, Eigen::Affine3d& transform) {

	transform = Eigen::Affine3d::Identity();
	std::ifstream in(filename.c_str());
	if (!in) {
		std::stringstream err;
		err << "Error loading transformation " << filename.c_str() << std::endl;
		std::cerr << err.str();
	}

	std::string line;
	for (int i = 0; i < 4; i++) {
		std::getline(in,line);

		std::istringstream sin(line);
		for (int j = 0; j < 4; j++) {
			sin >> transform (i,j);
		}
	}

	in.close();

}

void plotErrorGraph() {

	const std::string dataDir = programOptions.dataDir;//"../data/";

	int scanNo1 = 180;
	int step = programOptions.plotResultsStep;

	std::string gicpFileName = "%strans_gicp/trans_gicp_%d_%d";
	std::string resultFileName = "%strans_absolute/trans_result_%d_%d";
	std::string gtFileName = "%sTRANS_GT/trans_%d_%d";
	std::string fileString;
	int scanNo2 = 0;

	pcl::visualization::PCLPlotter *plotterTranslation, *plotterRotation;
	plotterTranslation = new pcl::visualization::PCLPlotter("Ground Truth Translation Plot");
	plotterTranslation->setXTitle("Scan Number");
	plotterTranslation->setYTitle("Mean Squared Translation Error");

	plotterRotation = new pcl::visualization::PCLPlotter("Ground Truth Rotation Plot");
	plotterRotation->setXTitle("Scan Difference");
	plotterRotation->setYTitle("Mean Squared Rotation Error");

	Eigen::Affine3d transformGicp, transformResult, transformGT;

	double txGt(0), tyGt(0), tzGt(0), rollGt(0), pitchGt(0), yawGt(0);
	double tx(0), ty(0), tz(0), roll(0), pitch(0), yaw(0);

	std::vector<std::pair<double, double> > errorTransGicp, errorRotGicp;
	std::vector<std::pair<double, double> > errorTransResult, errorRotResult;
	std::vector<std::pair<double, double> > errorTransBase, errorRotBase;

	int scanCounter = 0;
	while (scanNo1 < 990) {

		scanNo2 = scanNo1 + step;
		fileString = (boost::format(gtFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGT);

		svr_util::transform_get_rotation_from_affine(transformGT, &rollGt, &pitchGt, &yawGt);
		svr_util::transform_get_translation_from_affine(transformGT, &txGt, &tyGt, &tzGt);

		double errorTrans = svr_util::square(txGt) + svr_util::square(tyGt) + svr_util::square(tzGt);
		errorTrans = sqrt(errorTrans);
		errorTransBase.push_back(std::pair<double, double> (scanNo1, errorTrans));

		double errorRot = svr_util::square(rollGt) + svr_util::square(pitchGt) + svr_util::square(yawGt);
		errorRot = sqrt(errorRot);
		errorRotBase.push_back(std::pair<double, double> (scanNo1, errorRot));

		fileString = (boost::format(resultFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformResult);
		svr_util::transform_get_rotation_from_affine(transformResult, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformResult, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		errorTransResult.push_back(std::pair<double, double> (scanNo1, errorTrans));

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		errorRotResult.push_back(std::pair<double, double> (scanNo1, errorRot));

		fileString = (boost::format(gicpFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGicp);
		svr_util::transform_get_rotation_from_affine(transformGicp, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformGicp, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		errorTransGicp.push_back(std::pair<double, double> (scanNo1, errorTrans));

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		errorRotGicp.push_back(std::pair<double, double> (scanNo1, errorRot));

		//
		scanNo1 += 10;
		scanCounter++;
	}

	std::cout << "Total Scans: " << scanCounter << std::endl;

	plotterTranslation->addPlotData(errorTransBase, "Base Error");
	plotterTranslation->addPlotData(errorTransResult, "Result Error");
	plotterTranslation->addPlotData(errorTransGicp, "GICP Error");

	plotterRotation->addPlotData(errorRotBase, "Base Error");
	plotterRotation->addPlotData(errorRotResult, "Result Error");
	plotterRotation->addPlotData(errorRotGicp, "GICP Error");

	plotterRotation->plot();
	plotterTranslation->plot();

}

void plotGroundTruth() {

	const std::string dataDir = programOptions.dataDir;//"../data/";

	int scanNo1 = 180;
	int scanStepS = 5;
	int scanStepM = 10;
	int scanStepL = 15;
	int scanStepXL = 20;

	std::string gicpFileName = "%strans_gicp/trans_gicp_%d_%d";
	std::string resultFileName = "%strans_absolute/trans_result_%d_%d";
	std::string gtFileName = "%sTRANS_GT/trans_%d_%d";
	std::string fileString;
	int scanNo2 = 0;

	pcl::visualization::PCLPlotter *plotterTranslation, *plotterRotation;
	plotterTranslation = new pcl::visualization::PCLPlotter("Ground Truth Translation Plot");
	plotterTranslation->setXTitle("Scan Difference");
	plotterTranslation->setYTitle("Mean Squared Translation Error");

	plotterRotation = new pcl::visualization::PCLPlotter("Ground Truth Rotation Plot");
	plotterRotation->setXTitle("Scan Difference");
	plotterRotation->setYTitle("Mean Squared Rotation Error");

	double transGicpErrorS(0), rotGicpErrorS(0), transResultErrorS(0), rotResultErrorS(0), transBaseErrorS(0), rotBaseErrorS(0);
	double transGicpErrorM(0), rotGicpErrorM(0), transResultErrorM(0), rotResultErrorM(0), transBaseErrorM(0), rotBaseErrorM(0);
	double transGicpErrorL(0), rotGicpErrorL(0), transResultErrorL(0), rotResultErrorL(0), transBaseErrorL(0), rotBaseErrorL(0);
	double transGicpErrorXL(0), rotGicpErrorXL(0), transResultErrorXL(0), rotResultErrorXL(0), transBaseErrorXL(0), rotBaseErrorXL(0);

	Eigen::Affine3d transformGicp, transformResult, transformGT;

	double txGt(0), tyGt(0), tzGt(0), rollGt(0), pitchGt(0), yawGt(0);
	double tx(0), ty(0), tz(0), roll(0), pitch(0), yaw(0);

	std::vector<std::pair<double, double> > errorTransGicp, errorRotGicp;
	std::vector<std::pair<double, double> > errorTransResult, errorRotResult;
	std::vector<std::pair<double, double> > errorTransBase, errorRotBase;

	int scanCounter = 0;
	while (scanNo1 < 990) {

		// step = 5
		scanNo2 = scanNo1 + scanStepS;
		fileString = (boost::format(gtFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGT);

		svr_util::transform_get_rotation_from_affine(transformGT, &rollGt, &pitchGt, &yawGt);
		svr_util::transform_get_translation_from_affine(transformGT, &txGt, &tyGt, &tzGt);

		double errorTrans = svr_util::square(txGt) + svr_util::square(tyGt) + svr_util::square(tzGt);
		errorTrans = sqrt(errorTrans);
		transBaseErrorS += errorTrans;

		double errorRot = svr_util::square(rollGt) + svr_util::square(pitchGt) + svr_util::square(yawGt);
		errorRot = sqrt(errorRot);
		rotBaseErrorS += errorRot;

		fileString = (boost::format(resultFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformResult);
		svr_util::transform_get_rotation_from_affine(transformResult, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformResult, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transResultErrorS += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotResultErrorS += errorRot;

		fileString = (boost::format(gicpFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGicp);
		svr_util::transform_get_rotation_from_affine(transformGicp, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformGicp, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transGicpErrorS += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotGicpErrorS += errorRot;

		// step = 10

		scanNo2 = scanNo1 + scanStepM;
		fileString = (boost::format(gtFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGT);

		svr_util::transform_get_rotation_from_affine(transformGT, &rollGt, &pitchGt, &yawGt);
		svr_util::transform_get_translation_from_affine(transformGT, &txGt, &tyGt, &tzGt);

		errorTrans = svr_util::square(txGt) + svr_util::square(tyGt) + svr_util::square(tzGt);
		errorTrans = sqrt(errorTrans);
		transBaseErrorM += errorTrans;

		errorRot = svr_util::square(rollGt) + svr_util::square(pitchGt) + svr_util::square(yawGt);
		errorRot = sqrt(errorRot);
		rotBaseErrorM += errorRot;

		fileString = (boost::format(resultFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformResult);
		svr_util::transform_get_rotation_from_affine(transformResult, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformResult, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transResultErrorM += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotResultErrorM += errorRot;

		fileString = (boost::format(gicpFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGicp);
		svr_util::transform_get_rotation_from_affine(transformGicp, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformGicp, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transGicpErrorM += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotGicpErrorM += errorRot;

		// step = 15
		scanNo2 = scanNo1 + scanStepL;
		fileString = (boost::format(gtFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGT);

		svr_util::transform_get_rotation_from_affine(transformGT, &rollGt, &pitchGt, &yawGt);
		svr_util::transform_get_translation_from_affine(transformGT, &txGt, &tyGt, &tzGt);

		errorTrans = svr_util::square(txGt) + svr_util::square(tyGt) + svr_util::square(tzGt);
		errorTrans = sqrt(errorTrans);
		transBaseErrorL += errorTrans;

		errorRot = svr_util::square(rollGt) + svr_util::square(pitchGt) + svr_util::square(yawGt);
		errorRot = sqrt(errorRot);
		rotBaseErrorL += errorRot;

		fileString = (boost::format(resultFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformResult);
		svr_util::transform_get_rotation_from_affine(transformResult, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformResult, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transResultErrorL += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotResultErrorL += errorRot;

		fileString = (boost::format(gicpFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGicp);
		svr_util::transform_get_rotation_from_affine(transformGicp, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformGicp, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transGicpErrorL += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotGicpErrorL += errorRot;

		// step = 20
		scanNo2 = scanNo1 + scanStepXL;
		fileString = (boost::format(gtFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGT);

		svr_util::transform_get_rotation_from_affine(transformGT, &rollGt, &pitchGt, &yawGt);
		svr_util::transform_get_translation_from_affine(transformGT, &txGt, &tyGt, &tzGt);

		errorTrans = svr_util::square(txGt) + svr_util::square(tyGt) + svr_util::square(tzGt);
		errorTrans = sqrt(errorTrans);
		transBaseErrorXL += errorTrans;

		errorRot = svr_util::square(rollGt) + svr_util::square(pitchGt) + svr_util::square(yawGt);
		errorRot = sqrt(errorRot);
		rotBaseErrorXL += errorRot;

		fileString = (boost::format(resultFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformResult);
		svr_util::transform_get_rotation_from_affine(transformResult, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformResult, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transResultErrorXL += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotResultErrorXL += errorRot;

		fileString = (boost::format(gicpFileName)%dataDir%scanNo1%scanNo2).str();
		loadTransform(fileString, transformGicp);
		svr_util::transform_get_rotation_from_affine(transformGicp, &roll, &pitch, &yaw);
		svr_util::transform_get_translation_from_affine(transformGicp, &tx, &ty, &tz);

		errorTrans = svr_util::square(txGt-tx) + svr_util::square(tyGt-ty) + svr_util::square(tzGt-tz);
		errorTrans = sqrt(errorTrans);
		transGicpErrorXL += errorTrans;

		errorRot = svr_util::square(rollGt-roll) + svr_util::square(pitchGt-pitch) + svr_util::square(yawGt-yaw);
		errorRot = sqrt(errorRot);
		rotGicpErrorXL += errorRot;

		//
		scanNo1 += 10;
		scanCounter++;
	}

	std::cout << "Total Scans: " << scanCounter << std::endl;

	transBaseErrorS /= scanCounter;
	rotBaseErrorS /= scanCounter;
	transBaseErrorM /= scanCounter;
	rotBaseErrorM /= scanCounter;
	transBaseErrorL /= scanCounter;
	rotBaseErrorL /= scanCounter;
	transBaseErrorXL /= scanCounter;
	rotBaseErrorXL /= scanCounter;

	transGicpErrorS /= scanCounter;
	rotGicpErrorS /= scanCounter;
	transGicpErrorM /= scanCounter;
	rotGicpErrorM /= scanCounter;
	transGicpErrorL /= scanCounter;
	rotGicpErrorL /= scanCounter;
	transGicpErrorXL /= scanCounter;
	rotGicpErrorXL /= scanCounter;

	transResultErrorS /= scanCounter;
	rotResultErrorS /= scanCounter;
	transResultErrorM /= scanCounter;
	rotResultErrorM /= scanCounter;
	transResultErrorL /= scanCounter;
	rotResultErrorL /= scanCounter;
	transResultErrorXL /= scanCounter;
	rotResultErrorXL /= scanCounter;

	errorTransBase.push_back(std::pair<double, double>(5, transBaseErrorS));
	errorRotBase.push_back(std::pair<double, double>(5, rotBaseErrorS));
	errorTransBase.push_back(std::pair<double, double>(10, transBaseErrorM));
	errorRotBase.push_back(std::pair<double, double>(10, rotBaseErrorM));
	errorTransBase.push_back(std::pair<double, double>(15, transBaseErrorL));
	errorRotBase.push_back(std::pair<double, double>(15, rotBaseErrorL));
	errorTransBase.push_back(std::pair<double, double>(20, transBaseErrorXL));
	errorRotBase.push_back(std::pair<double, double>(20, rotBaseErrorXL));

	errorTransResult.push_back(std::pair<double, double>(5, transResultErrorS));
	errorRotResult.push_back(std::pair<double, double>(5, rotResultErrorS));
	errorTransResult.push_back(std::pair<double, double>(10, transResultErrorM));
	errorRotResult.push_back(std::pair<double, double>(10, rotResultErrorM));
	errorTransResult.push_back(std::pair<double, double>(15, transResultErrorL));
	errorRotResult.push_back(std::pair<double, double>(15, rotResultErrorL));
	errorTransResult.push_back(std::pair<double, double>(20, transResultErrorXL));
	errorRotResult.push_back(std::pair<double, double>(20, rotResultErrorXL));

	errorTransGicp.push_back(std::pair<double, double>(5, transGicpErrorS));
	errorRotGicp.push_back(std::pair<double, double>(5, rotGicpErrorS));
	errorTransGicp.push_back(std::pair<double, double>(10, transGicpErrorM));
	errorRotGicp.push_back(std::pair<double, double>(10, rotGicpErrorM));
	errorTransGicp.push_back(std::pair<double, double>(15, transGicpErrorL));
	errorRotGicp.push_back(std::pair<double, double>(15, rotGicpErrorL));
	errorTransGicp.push_back(std::pair<double, double>(20, transGicpErrorXL));
	errorRotGicp.push_back(std::pair<double, double>(20, rotGicpErrorXL));


	plotterTranslation->addPlotData(errorTransBase, "Base Error", vtkChart::POINTS);
	plotterTranslation->addPlotData(errorTransResult, "Result Error", vtkChart::POINTS);
	plotterTranslation->addPlotData(errorTransGicp, "GICP Error", vtkChart::POINTS);

	plotterRotation->addPlotData(errorRotBase, "Base Error", vtkChart::POINTS);
	plotterRotation->addPlotData(errorRotResult, "Result Error", vtkChart::POINTS);
	plotterRotation->addPlotData(errorRotGicp, "GICP Error", vtkChart::POINTS);

	plotterRotation->plot();
	plotterTranslation->plot();

}

int
main (int argc, char *argv[]) {

	if (initOptions(argc, argv))
		return 1;

	if (programOptions.plotAverage) {
		plotGroundTruth();
		return 0;
	} else if (programOptions.plotResultsStep > 0) {
		plotErrorGraph();
		return 0;
	}

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

		in.close();
	}

	svr::SupervoxelRegistration supervoxelRegistration (programOptions.vr, programOptions.sr);
	supervoxelRegistration.setDebug(programOptions.debug);
	supervoxelRegistration.setDebugScanString((boost::format("%d_%d")%s1%s2).str());
	supervoxelRegistration.setApproximate(programOptions.approx);

	if (programOptions.plotX || programOptions.plotY || programOptions.plotZ ||
			programOptions.plotRoll || programOptions.plotPitch || programOptions.plotYaw) {

		bool PLOT_OPTIMIZATION = programOptions.plotOptimizationIterations;

		scanB = temp;
		cout << "Plot Data" << std::endl;

		pcl::visualization::PCLPlotter* plotter;
		plotter = new pcl::visualization::PCLPlotter("cost plot");

		Eigen::Affine3d initialT = transform.inverse();
		supervoxelRegistration.setScans(scanA, scanB);
		supervoxelRegistration.preparePlotData();

		if (PLOT_OPTIMIZATION) {
			// plot Optimization
			if (programOptions.plotX)
				supervoxelRegistration.plotOptimizationIterationsForX(plotter, initialT, programOptions.plotStartTrans, programOptions.plotEndTrans, programOptions.plotStepTrans);
			if (programOptions.plotY)
				supervoxelRegistration.plotOptimizationIterationsForY(plotter, initialT, programOptions.plotStartTrans, programOptions.plotEndTrans, programOptions.plotStepTrans);
			if (programOptions.plotZ)
				supervoxelRegistration.plotOptimizationIterationsForZ(plotter, initialT, programOptions.plotStartTrans, programOptions.plotEndTrans, programOptions.plotStepTrans);
			if (programOptions.plotRoll)
				supervoxelRegistration.plotOptimizationIterationsForRoll(plotter, initialT, programOptions.plotStartAngle, programOptions.plotEndAngle, programOptions.plotStepAngle);
			if (programOptions.plotPitch)
				supervoxelRegistration.plotOptimizationIterationsForPitch(plotter, initialT, programOptions.plotStartAngle, programOptions.plotEndAngle, programOptions.plotStepAngle);
			if (programOptions.plotYaw)
				supervoxelRegistration.plotOptimizationIterationsForYaw(plotter, initialT, programOptions.plotStartAngle, programOptions.plotEndAngle, programOptions.plotStepAngle);
		} else {
			// plotting cost for different correspondences
			if (programOptions.plotX)
				supervoxelRegistration.plotCostFunctionForX(plotter, initialT, programOptions.plotStartTrans, programOptions.plotEndTrans, programOptions.plotStepTrans);
			if (programOptions.plotY)
				supervoxelRegistration.plotCostFunctionForY(plotter, initialT, programOptions.plotStartTrans, programOptions.plotEndTrans, programOptions.plotStepTrans);
			if (programOptions.plotZ)
				supervoxelRegistration.plotCostFunctionForZ(plotter, initialT, programOptions.plotStartTrans, programOptions.plotEndTrans, programOptions.plotStepTrans);
			if (programOptions.plotRoll)
				supervoxelRegistration.plotCostFunctionForRoll(plotter, initialT, programOptions.plotStartAngle, programOptions.plotEndAngle, programOptions.plotStepAngle);
			if (programOptions.plotPitch)
				supervoxelRegistration.plotCostFunctionForPitch(plotter, initialT, programOptions.plotStartAngle, programOptions.plotEndAngle, programOptions.plotStepAngle);
			if (programOptions.plotYaw)
				supervoxelRegistration.plotCostFunctionForYaw(plotter, initialT, programOptions.plotStartAngle, programOptions.plotEndAngle, programOptions.plotStepAngle);
		}
		// display
		std::string plotTitle;
		plotTitle = (boost::format("%d_%d")%s1%s2).str();

		plotter->setYTitle("Cost");
		plotter->setTitle(plotTitle.c_str());
		plotter->plot();

	} else if (programOptions.test != 0 || programOptions.showScans) {

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

		result = r.inverse();
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


