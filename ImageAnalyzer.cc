// -*- C++ -*-
//
// Package:    ImageAnalyzer
// Class:      ImageAnalyzer
// 
/**\class ImageAnalyzer ImageAnalyzer.cc Images/ImageAnalyzer/src/ImageAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Sun Apr 19 14:44:03 IST 2020
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "EgammaAnalysis/ElectronTools/interface/PFIsolationEstimator.h"
#include "EgammaAnalysis/ElectronTools/interface/EGammaCutBasedEleId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include <iostream>
#include "TFile.h"
#include "TH1.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include <string>
#include "TFile.h"
#include <vector>
#include "TBranch.h"
#include "TAxis.h"
#include "TCanvas.h"
#include "TH2.h"
#include "TGraph.h"
#include <stdio.h>
#include <stdlib.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TLorentzVector.h"
#include <fstream>
#include <iostream>



//
// class declaration
//

class ImageAnalyzer : public edm::EDAnalyzer {
   public:
      explicit ImageAnalyzer(const edm::ParameterSet&);
      ~ImageAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------


//
// constants, enums and typedefs
//

  struct Lepton{
    TLorentzVector v;
    int ind;};

  //asdasdasd

  ofstream passfile;
vector<Lepton> basic_ele;
vector<Lepton> medium_ele;
vector<Lepton> nonmed_ele;
};

//
// static data member definitions
//

//
// constructors and destructor
//
ImageAnalyzer::ImageAnalyzer(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

}


ImageAnalyzer::~ImageAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
ImageAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   PFIsolationEstimator isolator;
   isolator.initializeElectronIsolation(kTRUE);
   isolator.setConeSize(0.4);



#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif


   Handle<reco::VertexCollection> vertices;
   iEvent.getByLabel("offlinePrimaryVertices", vertices);
   LogInfo("Demo")<<"Number of vertices"<<vertices->size();
 

   Handle<reco::PFCandidateCollection> pfcollection;
   iEvent.getByLabel("particleFlow",pfcollection);


   Handle<reco::GsfElectronCollection> electroncollection;
   iEvent.getByLabel("gsfElectrons", electroncollection);
   LogInfo("Demo")<<"Electron Collection size "<<electroncollection->size();

   Handle<reco::ConversionCollection> convcollection;
   iEvent.getByLabel("allConversions", convcollection);

   Handle<reco::BeamSpot> beamSpot;
   iEvent.getByLabel("offlineBeamSpot", beamSpot);
   const reco::BeamSpot &beamspot= *(beamSpot.product());

   Handle<edm::SortedCollection<CaloTower,edm::StrictWeakOrdering<CaloTower>>>calotower;
   iEvent.getByLabel("towerMaker", calotower);

    edm::Handle<double> rho;
    iEvent.getByLabel("kt6PFJets","rho",rho);
    double rhoIso = *(rho.product());

    const reco::GsfElectronCollection ele_pro=*electroncollection.product();
    const reco::PFCandidateCollection* pf_pro=pfcollection.product();

basic_ele.clear();
medium_ele.clear();
nonmed_ele.clear();

int i=0;

for(reco::GsfElectronCollection::const_iterator electron=electroncollection->begin(); electron!=electroncollection->end(); ++electron){
  Lepton temp;
  temp.v.SetPtEtaPhiE(electron->pt(), electron->eta(), electron->phi(),
		      electron->energy());
  temp.ind = i;

  bool c1 = (electron->pt()>10) & (abs(electron->eta())<2.4);

  if(c1) basic_ele.push_back(temp);

  double iso_ch=0;
  double iso_em=0;
  double iso_nh=0;

  reco::GsfElectronRef ele(electroncollection,i);

  i=i+1;

  bool medium = EgammaCutBasedEleId::PassWP(EgammaCutBasedEleId::MEDIUM, ele, convcollection, beamspot, vertices, iso_ch, iso_em, iso_nh, rhoIso,ElectronEffectiveArea::kEleEAData2011);


  if (c1 & medium) medium_ele.push_back(temp);
  if(c1 & (!medium)) nonmed_ele.push_back(temp); }


int n = medium_ele.size();

for(int k=0; k<n; k++){
  TLorentzVector e1 = medium_ele.at(k).v;
  int index = medium_ele.at(k).ind;
  float eta = e1.Eta();
  float phi = e1.Phi();
  float pt = e1.Pt();
  float energy = e1.E();
  reco::GsfElectronRef ele(electroncollection, index);
  const reco::GsfElectron* ele_ptr=&ele_pro.at(index);
  reco::VertexRef vtx(vertices,0);
 
  float hcal1_et = ele->dr04HcalDepth1TowerSumEt();
  float hcal2_et = ele->dr04HcalDepth2TowerSumEt();
  float hcal1_etbc = ele->dr04HcalDepth1TowerSumEtBc();
  float hcal2_etbc = ele->dr04HcalDepth2TowerSumEtBc();
  float tksumpt = ele->dr04TkSumPt();
  //  float tksumptheep = ele->dr04TkSumPtHEEP();
  float ecal_rechit = ele->dr04EcalRecHitSumEt();
  float pfiso = isolator.fGetIsolation(ele_ptr, pf_pro, vtx, vertices);
  float chargediso = isolator.getIsolationCharged();
  float photoniso = isolator.getIsolationPhoton();
  float neutraliso = isolator.getIsolationNeutral();

  for(edm::SortedCollection<CaloTower, edm::StrictWeakOrdering<CaloTower>>::const_iterator calo=calotower->begin(); calo!=calotower->end();calo++){

    float calo_eta = calo->eta();
    float calo_phi = calo->phi();
    float calo_em = calo->emEnergy();
    float calo_had = calo->hadEnergy();
    float  calo_outer = calo->outerEnergy();
    float calo_emet = calo->emEt();
    float calo_hadet = calo->hadEt();
    float calo_outeret = calo->outerEt();

    if (fabs(sqrt(pow(eta-calo_eta,2) + pow(phi-calo_phi,2))) < 0.4){

      passfile<<pt<<" "<<eta<<" "<<phi<<" "<<energy<<" "<<pfiso<<" "<<chargediso<<" "<<photoniso<<" "<<neutraliso<<" "<<hcal1_et<<" "<<hcal2_et<<" "<<hcal1_etbc<<" "<<hcal2_etbc<<" "<<tksumpt<<" "<<ecal_rechit<<" "<<calo_eta<<" "<<calo_phi<<" "<<calo_em<<" "<<calo_had<<" "<<calo_outer<<" "<<calo_emet<<" "<<calo_hadet<<" "<<calo_outeret<<"\n";

    }
  }
 }



}


// ------------ method called once each job just before starting event loop  ------------
void ImageAnalyzer::beginJob()
{

  passfile.open("QCD_50/qcd1_83_fake.txt");

}

// ------------ method called once each job just after ending the event loop  ------------
void ImageAnalyzer::endJob() 
{

  passfile.close();
}

// ------------ method called when starting to processes a run  ------------
void 
ImageAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
ImageAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
ImageAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
ImageAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ImageAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ImageAnalyzer);
