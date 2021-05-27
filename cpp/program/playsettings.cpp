#include "../program/playsettings.h"

PlaySettings::PlaySettings()
  :initGamesWithPolicy(false),policyInitAreaProp(0.0),startPosesPolicyInitAreaProp(0.0),
   compensateAfterPolicyInitProb(0.0),sidePositionProb(0.0),
   policyInitAreaTemperature(1.0),handicapTemperature(1.0),
   compensateKomiVisits(20),estimateLeadVisits(10),estimateLeadProb(0.0),
   earlyForkGameProb(0.0),earlyForkGameExpectedMoveProp(0.0),forkGameProb(0.0),forkGameMinChoices(1),earlyForkGameMaxChoices(1),forkGameMaxChoices(1),
   sekiForkHackProb(0.0),fancyKomiVarying(false),
   cheapSearchProb(0),cheapSearchVisits(0),cheapSearchTargetWeight(0.0f),
   reduceVisits(false),reduceVisitsThreshold(100.0),reduceVisitsThresholdLookback(1),reducedVisitsMin(0),reducedVisitsWeight(1.0f),
   policySurpriseDataWeight(0.0),valueSurpriseDataWeight(0.0),scaleDataWeight(1.0),
   recordTreePositions(false),recordTreeThreshold(0),recordTreeTargetWeight(0.0f),
   noResolveTargetWeights(false),
   allowResignation(false),resignThreshold(0.0),resignConsecTurns(1),
   forSelfPlay(false),
   handicapAsymmetricPlayoutProb(0.0),normalAsymmetricPlayoutProb(0.0),maxAsymmetricRatio(2.0)
{}
PlaySettings::~PlaySettings()
{}

PlaySettings PlaySettings::loadForMatch(ConfigParser& cfg) {
  PlaySettings playSettings;
  playSettings.allowResignation = cfg.getBool("allowResignation");
  playSettings.resignThreshold = cfg.getDouble("resignThreshold",-1.0,0.0); //Threshold on [-1,1], regardless of winLossUtilityFactor
  playSettings.resignConsecTurns = cfg.getInt("resignConsecTurns",1,100);
  playSettings.compensateKomiVisits = cfg.contains("compensateKomiVisits") ? cfg.getInt("compensateKomiVisits",1,10000) : 100;
  return playSettings;
}

PlaySettings PlaySettings::loadForGatekeeper(ConfigParser& cfg) {
  PlaySettings playSettings;
  playSettings.allowResignation = cfg.getBool("allowResignation");
  playSettings.resignThreshold = cfg.getDouble("resignThreshold",-1.0,0.0); //Threshold on [-1,1], regardless of winLossUtilityFactor
  playSettings.resignConsecTurns = cfg.getInt("resignConsecTurns",1,100);
  playSettings.compensateKomiVisits = cfg.contains("compensateKomiVisits") ? cfg.getInt("compensateKomiVisits",1,10000) : 100;
  return playSettings;
}

PlaySettings PlaySettings::loadForSelfplay(ConfigParser& cfg) {
  PlaySettings playSettings;
  playSettings.initGamesWithPolicy = cfg.getBool("initGamesWithPolicy");
  playSettings.policyInitAreaProp = cfg.contains("policyInitAreaProp") ? cfg.getDouble("policyInitAreaProp",0.0,1.0) : 0.04;
  playSettings.startPosesPolicyInitAreaProp = cfg.contains("startPosesPolicyInitAreaProp") ? cfg.getDouble("startPosesPolicyInitAreaProp",0.0,1.0) : 0.0;
  playSettings.compensateAfterPolicyInitProb = cfg.getDouble("compensateAfterPolicyInitProb",0.0,1.0);
  playSettings.sidePositionProb =
    //forkSidePositionProb is the legacy name, included for backward compatibility
    (cfg.contains("forkSidePositionProb") && !cfg.contains("sidePositionProb")) ?
    cfg.getDouble("forkSidePositionProb",0.0,1.0) : cfg.getDouble("sidePositionProb",0.0,1.0);

  playSettings.policyInitAreaTemperature = cfg.contains("policyInitAreaTemperature") ? cfg.getDouble("policyInitAreaTemperature",0.1,5.0) : 1.0;
  playSettings.handicapTemperature = cfg.contains("handicapTemperature") ? cfg.getDouble("handicapTemperature",0.1,5.0) : 1.0;

  playSettings.compensateKomiVisits = cfg.contains("compensateKomiVisits") ? cfg.getInt("compensateKomiVisits",1,10000) : 20;
  playSettings.estimateLeadVisits = cfg.contains("estimateLeadVisits") ? cfg.getInt("estimateLeadVisits",1,10000) : 6;
  playSettings.estimateLeadProb = cfg.contains("estimateLeadProb") ? cfg.getDouble("estimateLeadProb",0.0,1.0) : 0.0;
  playSettings.fancyKomiVarying = cfg.contains("fancyKomiVarying") ? cfg.getBool("fancyKomiVarying") : false;

  playSettings.earlyForkGameProb = cfg.getDouble("earlyForkGameProb",0.0,0.5);
  playSettings.earlyForkGameExpectedMoveProp = cfg.getDouble("earlyForkGameExpectedMoveProp",0.0,1.0);
  playSettings.forkGameProb = cfg.getDouble("forkGameProb",0,0.5);
  playSettings.forkGameMinChoices = cfg.getInt("forkGameMinChoices",1,100);
  playSettings.earlyForkGameMaxChoices = cfg.getInt("earlyForkGameMaxChoices",1,100);
  playSettings.forkGameMaxChoices = cfg.getInt("forkGameMaxChoices",1,100);
  playSettings.cheapSearchProb = cfg.getDouble("cheapSearchProb",0.0,1.0);
  playSettings.cheapSearchVisits = cfg.getInt("cheapSearchVisits",1,10000000);
  playSettings.cheapSearchTargetWeight = cfg.getFloat("cheapSearchTargetWeight",0.0f,1.0f);
  playSettings.reduceVisits = cfg.getBool("reduceVisits");
  playSettings.reduceVisitsThreshold = cfg.getDouble("reduceVisitsThreshold",0.0,0.999999);
  playSettings.reduceVisitsThresholdLookback = cfg.getInt("reduceVisitsThresholdLookback",0,1000);
  playSettings.reducedVisitsMin = cfg.getInt("reducedVisitsMin",1,10000000);
  playSettings.reducedVisitsWeight = cfg.getFloat("reducedVisitsWeight",0.0f,1.0f);
  playSettings.policySurpriseDataWeight = cfg.getDouble("policySurpriseDataWeight",0.0,1.0);
  playSettings.valueSurpriseDataWeight = cfg.getDouble("valueSurpriseDataWeight",0.0,1.0);
  playSettings.scaleDataWeight = cfg.contains("scaleDataWeight") ? cfg.getDouble("scaleDataWeight",0.01,10.0) : 1.0;
  playSettings.handicapAsymmetricPlayoutProb = cfg.getDouble("handicapAsymmetricPlayoutProb",0.0,1.0);
  playSettings.normalAsymmetricPlayoutProb = cfg.getDouble("normalAsymmetricPlayoutProb",0.0,1.0);
  playSettings.maxAsymmetricRatio = cfg.getDouble("maxAsymmetricRatio",1.0,100.0);
  playSettings.minAsymmetricCompensateKomiProb = cfg.getDouble("minAsymmetricCompensateKomiProb",0.0,1.0);
  playSettings.sekiForkHackProb = cfg.contains("sekiForkHackProb") ? cfg.getDouble("sekiForkHackProb",0.0,1.0) : 0.0;
  playSettings.forSelfPlay = true;

  if(playSettings.policySurpriseDataWeight + playSettings.valueSurpriseDataWeight > 1.0)
    throw StringError("policySurpriseDataWeight + valueSurpriseDataWeight > 1.0");

  return playSettings;
}
