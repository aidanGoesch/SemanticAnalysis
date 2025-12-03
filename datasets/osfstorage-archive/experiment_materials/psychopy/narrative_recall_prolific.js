/********************************** 
 * Narrative_Recall_Prolific Test *
 **********************************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2022.2.4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'narrative_recall_prolific';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(WelcomeRoutineBegin());
flowScheduler.add(WelcomeRoutineEachFrame());
flowScheduler.add(WelcomeRoutineEnd());
flowScheduler.add(Survey_prolific_IDRoutineBegin());
flowScheduler.add(Survey_prolific_IDRoutineEachFrame());
flowScheduler.add(Survey_prolific_IDRoutineEnd());
flowScheduler.add(informedconsent_1RoutineBegin());
flowScheduler.add(informedconsent_1RoutineEachFrame());
flowScheduler.add(informedconsent_1RoutineEnd());
flowScheduler.add(informedconsent_2RoutineBegin());
flowScheduler.add(informedconsent_2RoutineEachFrame());
flowScheduler.add(informedconsent_2RoutineEnd());
flowScheduler.add(informedconsent_3RoutineBegin());
flowScheduler.add(informedconsent_3RoutineEachFrame());
flowScheduler.add(informedconsent_3RoutineEnd());
flowScheduler.add(informedconsent_4RoutineBegin());
flowScheduler.add(informedconsent_4RoutineEachFrame());
flowScheduler.add(informedconsent_4RoutineEnd());
flowScheduler.add(informedconsent_5RoutineBegin());
flowScheduler.add(informedconsent_5RoutineEachFrame());
flowScheduler.add(informedconsent_5RoutineEnd());
flowScheduler.add(soundCheckRoutineBegin());
flowScheduler.add(soundCheckRoutineEachFrame());
flowScheduler.add(soundCheckRoutineEnd());
flowScheduler.add(verbal_instructionsRoutineBegin());
flowScheduler.add(verbal_instructionsRoutineEachFrame());
flowScheduler.add(verbal_instructionsRoutineEnd());
flowScheduler.add(task_instructionsRoutineBegin());
flowScheduler.add(task_instructionsRoutineEachFrame());
flowScheduler.add(task_instructionsRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);
flowScheduler.add(Survey_Likert_Q1RoutineBegin());
flowScheduler.add(Survey_Likert_Q1RoutineEachFrame());
flowScheduler.add(Survey_Likert_Q1RoutineEnd());
flowScheduler.add(Survey_Likert_Q2RoutineBegin());
flowScheduler.add(Survey_Likert_Q2RoutineEachFrame());
flowScheduler.add(Survey_Likert_Q2RoutineEnd());
flowScheduler.add(Survey_Likert_Q3RoutineBegin());
flowScheduler.add(Survey_Likert_Q3RoutineEachFrame());
flowScheduler.add(Survey_Likert_Q3RoutineEnd());
flowScheduler.add(Survey_OpenEnded_Q1RoutineBegin());
flowScheduler.add(Survey_OpenEnded_Q1RoutineEachFrame());
flowScheduler.add(Survey_OpenEnded_Q1RoutineEnd());
flowScheduler.add(Survey_OpenEnded_Q2RoutineBegin());
flowScheduler.add(Survey_OpenEnded_Q2RoutineEachFrame());
flowScheduler.add(Survey_OpenEnded_Q2RoutineEnd());
flowScheduler.add(Survey_OpenEnded_Q3RoutineBegin());
flowScheduler.add(Survey_OpenEnded_Q3RoutineEachFrame());
flowScheduler.add(Survey_OpenEnded_Q3RoutineEnd());
flowScheduler.add(Survey_OpenEnded_Q4RoutineBegin());
flowScheduler.add(Survey_OpenEnded_Q4RoutineEachFrame());
flowScheduler.add(Survey_OpenEnded_Q4RoutineEnd());
flowScheduler.add(Survey_OpenEnded_Q5RoutineBegin());
flowScheduler.add(Survey_OpenEnded_Q5RoutineEachFrame());
flowScheduler.add(Survey_OpenEnded_Q5RoutineEnd());
flowScheduler.add(Survey_OpenEnded_Q6RoutineBegin());
flowScheduler.add(Survey_OpenEnded_Q6RoutineEachFrame());
flowScheduler.add(Survey_OpenEnded_Q6RoutineEnd());
flowScheduler.add(debreifRoutineBegin());
flowScheduler.add(debreifRoutineEachFrame());
flowScheduler.add(debreifRoutineEnd());
flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    {'name': 'sound_check/sound_check.mp3', 'path': 'sound_check/sound_check.mp3'},
    {'name': 'consent/Prolific_Consent_PART_2.png', 'path': 'consent/Prolific_Consent_PART_2.png'},
    {'name': 'consent/Prolific_Consent_PART_3.png', 'path': 'consent/Prolific_Consent_PART_3.png'},
    {'name': 'conditions.xlsx', 'path': 'conditions.xlsx'},
    {'name': 'consent/Prolific_Consent_PART_4.png', 'path': 'consent/Prolific_Consent_PART_4.png'},
    {'name': 'consent/Prolific_Consent_PART_5.png', 'path': 'consent/Prolific_Consent_PART_5.png'},
    {'name': 'consent/Prolific_Consent_PART_1.png', 'path': 'consent/Prolific_Consent_PART_1.png'},
    {'name': 'stories/baseball_audio.mp3', 'path': 'stories/baseball_audio.mp3'},
    {'name': 'consent/Click_to_continue.png', 'path': 'consent/Click_to_continue.png'},
    {'name': 'stories/oregontrail_audio.mp3', 'path': 'stories/oregontrail_audio.mp3'},
    {'name': 'sound_check/verbal_instructions.mp3', 'path': 'sound_check/verbal_instructions.mp3'},
    {'name': 'consent/NYUlogo.PNG', 'path': 'consent/NYUlogo.PNG'}
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2022.2.4';
  expInfo['OS'] = window.navigator.platform;

  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);

  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  
  return Scheduler.Event.NEXT;
}


var WelcomeClock;
var bakground1_2;
var logo1;
var AgeQuestion;
var chrome_rec;
var key_18yoResp;
var Survey_prolific_IDClock;
var text_31;
var text_32;
var textbox_8;
var key_resp_9;
var informedconsent_1Clock;
var bakground1_4;
var consent3;
var text_4;
var key_18yoResp_3;
var informedconsent_2Clock;
var bakground1_5;
var consent3_2;
var text_5;
var key_18yoResp_4;
var informedconsent_3Clock;
var bakground1_6;
var consent3_3;
var text_6;
var key_18yoResp_5;
var informedconsent_4Clock;
var bakground1_7;
var consent3_4;
var text_29;
var key_18yoResp_6;
var informedconsent_5Clock;
var bakground1_8;
var consent3_5;
var text_30;
var key_18yoResp_7;
var soundCheckClock;
var heard_sound;
var text_2;
var mic_check;
var sound_1;
var verbal_instructionsClock;
var text_7;
var sound_2;
var task_instructionsClock;
var general_instructions;
var continue_space;
var narrativeExposureClock;
var text;
var story_1;
var recall_instructionsClock;
var instruction;
var continue_space_2;
var verbalRecallClock;
var recording_in_progress;
var recall_instruction;
var polygon;
var key_resp_2;
var polygon_2;
var mic_2;
var next_story_instuctClock;
var general_instructions_2;
var continue_space_3;
var Survey_Likert_Q1Clock;
var text_17;
var text_18;
var slider_4;
var image;
var mouse;
var Survey_Likert_Q2Clock;
var text_11;
var text_12;
var slider_2;
var image_2;
var mouse_2;
var Survey_Likert_Q3Clock;
var text_13;
var text_14;
var slider_3;
var image_3;
var mouse_3;
var Survey_OpenEnded_Q1Clock;
var text_15;
var text_16;
var textbox_2;
var key_resp_3;
var Survey_OpenEnded_Q2Clock;
var text_19;
var text_20;
var textbox_3;
var key_resp_4;
var Survey_OpenEnded_Q3Clock;
var text_21;
var text_22;
var textbox_4;
var key_resp_5;
var Survey_OpenEnded_Q4Clock;
var text_23;
var text_24;
var textbox_5;
var key_resp_6;
var Survey_OpenEnded_Q5Clock;
var text_25;
var text_26;
var textbox_6;
var key_resp_7;
var Survey_OpenEnded_Q6Clock;
var text_27;
var text_28;
var textbox_7;
var key_resp_8;
var debreifClock;
var text_3;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "Welcome"
  WelcomeClock = new util.Clock();
  bakground1_2 = new visual.Rect ({
    win: psychoJS.window, name: 'bakground1_2', 
    width: [3, 3][0], height: [3, 3][1],
    ori: 0, pos: [0, 0],
    lineWidth: 1, 
    colorSpace: 'rgb',
    lineColor: new util.Color([1.0, 1.0, 1.0]),
    fillColor: new util.Color([1.0, 1.0, 1.0]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  logo1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'logo1', units : undefined, 
    image : 'consent/NYUlogo.PNG', mask : undefined,
    ori : 0, pos : [0, 0.35], size : [0.2, 0.2],
    color : new util.Color([1, 1, 1]), opacity : 1,
    flipHoriz : false, flipVert : false,
    texRes : 128, interpolate : true, depth : -1.0 
  });
  AgeQuestion = new visual.TextStim({
    win: psychoJS.window,
    name: 'AgeQuestion',
    text: 'Welcome to this study!\n\nPress any key to continue.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: 1,
    depth: -2.0 
  });
  
  chrome_rec = new visual.TextStim({
    win: psychoJS.window,
    name: 'chrome_rec',
    text: '***You will need Chrome to run this experiment. If you are not using Chrome, please exit and launch again in Chrome.***\n\n***Do not press ESC unless you want to leave this study without completing it!**',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.3)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: 1.0,
    depth: -3.0 
  });
  
  key_18yoResp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_prolific_ID"
  Survey_prolific_IDClock = new util.Clock();
  text_31 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_31',
    text: 'What is your Prolific ID?',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_32 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_32',
    text: "[Please type below to response to the question above. Once you are done with your response, please press 'space' on your keyboard to move on.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_8 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_8',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_9 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "informedconsent_1"
  informedconsent_1Clock = new util.Clock();
  bakground1_4 = new visual.Rect ({
    win: psychoJS.window, name: 'bakground1_4', 
    width: [3, 3][0], height: [3, 3][1],
    ori: 0, pos: [0, 0],
    lineWidth: 1, 
    colorSpace: 'rgb',
    lineColor: new util.Color([1.0, 1.0, 1.0]),
    fillColor: new util.Color([1.0, 1.0, 1.0]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  consent3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'consent3', units : undefined, 
    image : 'consent/Prolific_Consent_PART_1.png', mask : undefined,
    ori : 0.0, pos : [0, 0], size : [1.1, 0.78],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  text_4 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_4',
    text: 'Please read the consent form carefully and press [a] to move to the next page of the form (pg. 1/5).',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.42], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: undefined,
    depth: -2.0 
  });
  
  key_18yoResp_3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "informedconsent_2"
  informedconsent_2Clock = new util.Clock();
  bakground1_5 = new visual.Rect ({
    win: psychoJS.window, name: 'bakground1_5', 
    width: [3, 3][0], height: [3, 3][1],
    ori: 0, pos: [0, 0],
    lineWidth: 1, 
    colorSpace: 'rgb',
    lineColor: new util.Color([1.0, 1.0, 1.0]),
    fillColor: new util.Color([1.0, 1.0, 1.0]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  consent3_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'consent3_2', units : undefined, 
    image : 'consent/Prolific_Consent_PART_2.png', mask : undefined,
    ori : 0.0, pos : [0, 0], size : [1.1, 0.78],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  text_5 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_5',
    text: 'Please read the consent form carefully and press [a] to move to the next page of the form (pg. 2/5)',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.42], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: undefined,
    depth: -2.0 
  });
  
  key_18yoResp_4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "informedconsent_3"
  informedconsent_3Clock = new util.Clock();
  bakground1_6 = new visual.Rect ({
    win: psychoJS.window, name: 'bakground1_6', 
    width: [3, 3][0], height: [3, 3][1],
    ori: 0, pos: [0, 0],
    lineWidth: 1, 
    colorSpace: 'rgb',
    lineColor: new util.Color([1.0, 1.0, 1.0]),
    fillColor: new util.Color([1.0, 1.0, 1.0]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  consent3_3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'consent3_3', units : undefined, 
    image : 'consent/Prolific_Consent_PART_3.png', mask : undefined,
    ori : 0.0, pos : [0, 0], size : [1.1, 0.78],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  text_6 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_6',
    text: 'Please read the consent form carefully and press [a] to move to the next page of the form (pg. 3/5)',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.42], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: undefined,
    depth: -2.0 
  });
  
  key_18yoResp_5 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "informedconsent_4"
  informedconsent_4Clock = new util.Clock();
  bakground1_7 = new visual.Rect ({
    win: psychoJS.window, name: 'bakground1_7', 
    width: [3, 3][0], height: [3, 3][1],
    ori: 0, pos: [0, 0],
    lineWidth: 1, 
    colorSpace: 'rgb',
    lineColor: new util.Color([1.0, 1.0, 1.0]),
    fillColor: new util.Color([1.0, 1.0, 1.0]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  consent3_4 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'consent3_4', units : undefined, 
    image : 'consent/Prolific_Consent_PART_4.png', mask : undefined,
    ori : 0.0, pos : [0, (- 0.13)], size : [1, 0.73],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  text_29 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_29',
    text: 'Please read the consent form (pg. 4/5) carefully.\n\nPlease press [s] if you agree to the authorization statement below. This confirmation is equivalent to signing and dating the document. \n\nOtherwise, please exit the experiment now.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.35], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: undefined,
    depth: -2.0 
  });
  
  key_18yoResp_6 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "informedconsent_5"
  informedconsent_5Clock = new util.Clock();
  bakground1_8 = new visual.Rect ({
    win: psychoJS.window, name: 'bakground1_8', 
    width: [3, 3][0], height: [3, 3][1],
    ori: 0, pos: [0, 0],
    lineWidth: 1, 
    colorSpace: 'rgb',
    lineColor: new util.Color([1.0, 1.0, 1.0]),
    fillColor: new util.Color([1.0, 1.0, 1.0]),
    opacity: 1, depth: 0, interpolate: true,
  });
  
  consent3_5 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'consent3_5', units : undefined, 
    image : 'consent/Prolific_Consent_PART_5.png', mask : undefined,
    ori : 0.0, pos : [0, (- 0.15)], size : [1, 0.73],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  text_30 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_30',
    text: 'Please read the consent form (pg. 5/5) carefully. \n\nPlease press [s] if you agree to the statement below and cofirm that you are at least 18 years of age. This confirmation is equivalent to signing and dating the document.\n\nIf you do not agree to participate or are not  at least 18 years of age, please exit the experiment now.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.35], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('black'),  opacity: undefined,
    depth: -2.0 
  });
  
  key_18yoResp_7 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "soundCheck"
  soundCheckClock = new util.Clock();
  heard_sound = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  text_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_2',
    text: 'This experiment requires you to listen to stories. Let us test your sound output. You may use your computer speaker, headphones or earphones.\n\nThis page is also testing your microphone. Please enable your microphone for this website if you are prompted to do so by your browser.\n\nCan you clearly hear a constant sound? If so, please press [spacebar] to continue. If you do not hear a sound, please make sure your volume is turned on and/or check your audio settings. ',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  mic_check = new sound.Microphone({
    win : psychoJS.window, 
    name:'mic_check',
    sampleRateHz : 16000,
    channels : 'auto',
    maxRecordingSize : 24000.0,
    loopback : true,
    policyWhenFull : 'ignore',
  });
  sound_1 = new sound.Sound({
    win: psychoJS.window,
    value: 'sound_check/sound_check.mp3',
    secs: (- 1),
    });
  sound_1.setVolume(1.0);
  // Initialize components for Routine "verbal_instructions"
  verbal_instructionsClock = new util.Clock();
  text_7 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_7',
    text: '[Verbal Instructions]\n\nMake sure your volume is turned on. These verbal instructions will be followed by written instructions. Please be sure to pay close attention to both sets of instructions and any other written instructions throughout the task.\n\nAfter listening to these instructions, the experiment will continue automatically. ',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  sound_2 = new sound.Sound({
    win: psychoJS.window,
    value: 'sound_check/verbal_instructions.mp3',
    secs: (- 1),
    });
  sound_2.setVolume(1.0);
  // Initialize components for Routine "task_instructions"
  task_instructionsClock = new util.Clock();
  general_instructions = new visual.TextStim({
    win: psychoJS.window,
    name: 'general_instructions',
    text: 'PLEASE READ THESE INSTRUCTIONS CAREFULLY\n\nIn this experiment, you will listen to two stories which range from 7 minutes to 15 minutes. After each story, you will be asked to recollect the story in as much detail as possible by speaking aloud. Your voice will be automatically recorded by your computer microphone. We ask that you describe the story in as much detail as you can and try to do so in the order in which you heard the story. Remembering the entire story in as much detail as possible will be more important than remembering the correct order. Therefore, feel free to return to any point you may have missed when recalling the story. Please do not engage in any other tasks while listening to the story, such as taking notes or texting.\n\nThis data will be used for scientific purposes and we rely on your attention and diligent participation throughout the experiment. Thank you.\n\nPlease press the [s] to hear the first story.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.02,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  continue_space = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "narrativeExposure"
  narrativeExposureClock = new util.Clock();
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: 'Please listen carefully!',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  story_1 = new sound.Sound({
    win: psychoJS.window,
    value: 'A',
    secs: (- 1),
    });
  story_1.setVolume(1.0);
  // Initialize components for Routine "recall_instructions"
  recall_instructionsClock = new util.Clock();
  instruction = new visual.TextStim({
    win: psychoJS.window,
    name: 'instruction',
    text: 'PLEASE READ THESE INSTRUCTIONS CAREFULLY\n\nNext, we ask that you recall the story in as much detail as possible. We ask that you speak for at least 4 minutes, but longer is better. A green circle on the screen will turn yellow when 4 minutes have elapsed, but please continue to recall until you complete the story.\n\nPlease try to recall the story in as much detail as possible and in the correct order. Please return to any point you may have missed. Your voice will be automatically recorded using your computer’s microphone. \n\nOnce 4 minutes have passed AND you have completed recalling the complete story, you will say “I am done”, and then press [d] to complete your recording.\n\nPlease press the [spacebar] to begin recalling the story.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  continue_space_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "verbalRecall"
  verbalRecallClock = new util.Clock();
  recording_in_progress = new visual.TextStim({
    win: psychoJS.window,
    name: 'recording_in_progress',
    text: 'RECORDING IN PROGRESS',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.4], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color([0.0, 0.0, 0.0]),  opacity: undefined,
    depth: 0.0 
  });
  
  recall_instruction = new visual.TextStim({
    win: psychoJS.window,
    name: 'recall_instruction',
    text: 'Please recall the story in as much detail as possible.\n\nYou must recall the story for at least four minutes (the green circle will turn yellow once two minutes have elapsed), but please continue beyond this time to complete your recollection of the story. Please say "I am done" and then press [\'d\'] when you are done recalling the entire story.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  polygon = new visual.Polygon({
    win: psychoJS.window, name: 'polygon', 
    edges: 100, size:[0.15, 0.15],
    ori: 0.0, pos: [0, (- 0.4)],
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color(undefined),
    fillColor: new util.Color('green'),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  key_resp_2 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  polygon_2 = new visual.Polygon({
    win: psychoJS.window, name: 'polygon_2', 
    edges: 100, size:[0.15, 0.15],
    ori: 0.0, pos: [0, (- 0.4)],
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('yellow'),
    opacity: undefined, depth: -4, interpolate: true,
  });
  
  mic_2 = new sound.Microphone({
    win : psychoJS.window, 
    name:'mic_2',
    sampleRateHz : 44100,
    channels : 'auto',
    maxRecordingSize : 24000.0,
    loopback : true,
    policyWhenFull : 'ignore',
  });
  // Initialize components for Routine "next_story_instuct"
  next_story_instuctClock = new util.Clock();
  general_instructions_2 = new visual.TextStim({
    win: psychoJS.window,
    name: 'general_instructions_2',
    text: 'Thank you!\n\nIF this is the first story you heard, you will now perform the same task but with a different story. Please listen carefully as you will once again be tested for your memory of the story.\n\nOtherwise, you will be given about 10 survey question before completing the experiment.\n\nPress the [r] to continue.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  continue_space_3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_Likert_Q1"
  Survey_Likert_Q1Clock = new util.Clock();
  text_17 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_17',
    text: 'I found the experiment difficult.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.055,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_18 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_18',
    text: '[Please use you mouse to select a point (out of the five options) on the slider that best indicates your response to the statement above. Once you respond, a button will appear which you may click to move to the next question. Feel free to change your answer before moving on.]',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.025,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  slider_4 = new visual.Slider({
    win: psychoJS.window, name: 'slider_4',
    startValue: undefined,
    size: [1.0, 0.1], pos: [0, (- 0.2)], ori: 0.0, units: 'height',
    labels: ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"], fontSize: 0.025, ticks: [1, 2, 3, 4, 5],
    granularity: 1.0, style: ["RATING"],
    color: new util.Color('LightGray'), markerColor: new util.Color('Red'), lineColor: new util.Color('White'), 
    opacity: undefined, fontFamily: 'Open Sans', bold: true, italic: false, depth: -2, 
    flip: false,
  });
  
  image = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image', units : undefined, 
    image : 'consent/Click_to_continue.png', mask : undefined,
    ori : 0.0, pos : [0.45, (- 0.45)], size : [0.3, 0.1],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  mouse = new core.Mouse({
    win: psychoJS.window,
  });
  mouse.mouseClock = new util.Clock();
  // Initialize components for Routine "Survey_Likert_Q2"
  Survey_Likert_Q2Clock = new util.Clock();
  text_11 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_11',
    text: 'I understood the task instructions.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.055,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_12 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_12',
    text: '[Please use you mouse to select a point (out of the five options) on the slider that best indicates your response to the statement above. Once you respond, a button will appear which you may click to move to the next question. Feel free to change your answer before moving on.]',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.025,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  slider_2 = new visual.Slider({
    win: psychoJS.window, name: 'slider_2',
    startValue: undefined,
    size: [1.0, 0.1], pos: [0, (- 0.2)], ori: 0.0, units: 'height',
    labels: ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"], fontSize: 0.025, ticks: [1, 2, 3, 4, 5],
    granularity: 1.0, style: ["RATING"],
    color: new util.Color('LightGray'), markerColor: new util.Color('Red'), lineColor: new util.Color('White'), 
    opacity: undefined, fontFamily: 'Open Sans', bold: true, italic: false, depth: -2, 
    flip: false,
  });
  
  image_2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_2', units : undefined, 
    image : 'consent/Click_to_continue.png', mask : undefined,
    ori : 0.0, pos : [0.45, (- 0.45)], size : [0.3, 0.1],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  mouse_2 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_2.mouseClock = new util.Clock();
  // Initialize components for Routine "Survey_Likert_Q3"
  Survey_Likert_Q3Clock = new util.Clock();
  text_13 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_13',
    text: 'I was engaged in the experiment. In other words, I payed close attention to the stories and took the task seriously [Please answer honestly, this will not affect your payment]',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.035,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_14 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_14',
    text: '[Please use you mouse to select a point (out of the five options) on the slider that best indicates your response to the statement above. Once you respond, a button will appear which you may click to move to the next question. Feel free to change your answer as you wish before moving on.]',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.025,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  slider_3 = new visual.Slider({
    win: psychoJS.window, name: 'slider_3',
    startValue: undefined,
    size: [1.0, 0.1], pos: [0, (- 0.2)], ori: 0.0, units: 'height',
    labels: ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"], fontSize: 0.025, ticks: [1, 2, 3, 4, 5],
    granularity: 1.0, style: ["RATING"],
    color: new util.Color('LightGray'), markerColor: new util.Color('Red'), lineColor: new util.Color('White'), 
    opacity: undefined, fontFamily: 'Open Sans', bold: true, italic: false, depth: -2, 
    flip: false,
  });
  
  image_3 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image_3', units : undefined, 
    image : 'consent/Click_to_continue.png', mask : undefined,
    ori : 0.0, pos : [0.45, (- 0.45)], size : [0.3, 0.1],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -3.0 
  });
  mouse_3 = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_3.mouseClock = new util.Clock();
  // Initialize components for Routine "Survey_OpenEnded_Q1"
  Survey_OpenEnded_Q1Clock = new util.Clock();
  text_15 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_15',
    text: 'Did you use a particular strategy to memorize the story (e.g., imagining yourself in the story or simply paying close attention)?',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_16 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_16',
    text: "[Please type below to response to the question above. Once you are done with your response, please press '9' on your keyboard to move on to the next question.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_2 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_2',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_3 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_OpenEnded_Q2"
  Survey_OpenEnded_Q2Clock = new util.Clock();
  text_19 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_19',
    text: 'Were there any moments in either story that you found particularly memorable? If so, please describe them below. ',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_20 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_20',
    text: "[Please type below to response to the question above. Once you are done with your response, please press '9' on your keyboard to move on to the next question.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_3 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_3',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_4 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_OpenEnded_Q3"
  Survey_OpenEnded_Q3Clock = new util.Clock();
  text_21 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_21',
    text: 'Is there anything you think we should know about your experience taking our experiment?',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_22 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_22',
    text: "[Please type below to response to the question above. Once you are done with your response, please press '9' on your keyboard to move on to the next question.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_4 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_4',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_5 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_OpenEnded_Q4"
  Survey_OpenEnded_Q4Clock = new util.Clock();
  text_23 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_23',
    text: 'Are you a native english speaker? That is, is english your first language?',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_24 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_24',
    text: "[Please type below to response to the question above. Once you are done with your response, please press '9' on your keyboard to move on to the next question.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_5 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_5',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_6 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_OpenEnded_Q5"
  Survey_OpenEnded_Q5Clock = new util.Clock();
  text_25 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_25',
    text: 'Do you speak any other languages besides English? If so, how many years have you spoken this language(s) and what is your level of fluency.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_26 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_26',
    text: "[Please type below to response to the question above. Once you are done with your response, please press '9' on your keyboard to move on to the next question.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_6 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_6',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_7 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "Survey_OpenEnded_Q6"
  Survey_OpenEnded_Q6Clock = new util.Clock();
  text_27 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_27',
    text: 'What were you doing while listening to the stories (e.g., only listening, browsing the internet, taking notes, etc.)? [Please answer honestly, your answer will not affect your payment]',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.25], height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  text_28 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_28',
    text: "[Please type below to response to the question above. Once you are done with your response, please press '9' on your keyboard to move on to the next question.]",
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0.1], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -1.0 
  });
  
  textbox_7 = new visual.TextBox({
    win: psychoJS.window,
    name: 'textbox_7',
    text: 'DELETE THIS TEXT AND TYPE HERE',
    font: 'Open Sans',
    pos: [0, (- 0.2)], letterHeight: 0.03,
    size: [null, null],  units: undefined, 
    color: 'white', colorSpace: 'rgb',
    fillColor: undefined, borderColor: undefined,
    languageStyle: 'LTR',
    bold: false, italic: false,
    opacity: undefined,
    padding: 0.0,
    alignment: 'center',
    editable: true,
    multiline: true,
    anchor: 'center',
    depth: -2.0 
  });
  
  key_resp_8 = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  // Initialize components for Routine "debreif"
  debreifClock = new util.Clock();
  text_3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_3',
    text: 'Thank you for participanting in the experiment!\nYour completion code is CT6CIC4T\n\nplease press ESC to exit the experiment.',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, 0], height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var _key_18yoResp_allKeys;
var WelcomeComponents;
function WelcomeRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Welcome' ---
    t = 0;
    WelcomeClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_18yoResp.keys = undefined;
    key_18yoResp.rt = undefined;
    _key_18yoResp_allKeys = [];
    // keep track of which components have finished
    WelcomeComponents = [];
    WelcomeComponents.push(bakground1_2);
    WelcomeComponents.push(logo1);
    WelcomeComponents.push(AgeQuestion);
    WelcomeComponents.push(chrome_rec);
    WelcomeComponents.push(key_18yoResp);
    
    for (const thisComponent of WelcomeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function WelcomeRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Welcome' ---
    // get current time
    t = WelcomeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *bakground1_2* updates
    if (t >= 0.0 && bakground1_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      bakground1_2.tStart = t;  // (not accounting for frame time here)
      bakground1_2.frameNStart = frameN;  // exact frame index
      
      bakground1_2.setAutoDraw(true);
    }

    
    // *logo1* updates
    if (t >= 0.0 && logo1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      logo1.tStart = t;  // (not accounting for frame time here)
      logo1.frameNStart = frameN;  // exact frame index
      
      logo1.setAutoDraw(true);
    }

    
    // *AgeQuestion* updates
    if (t >= 0.0 && AgeQuestion.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      AgeQuestion.tStart = t;  // (not accounting for frame time here)
      AgeQuestion.frameNStart = frameN;  // exact frame index
      
      AgeQuestion.setAutoDraw(true);
    }

    
    // *chrome_rec* updates
    if (t >= 0.0 && chrome_rec.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      chrome_rec.tStart = t;  // (not accounting for frame time here)
      chrome_rec.frameNStart = frameN;  // exact frame index
      
      chrome_rec.setAutoDraw(true);
    }

    
    // *key_18yoResp* updates
    if (t >= 0.0 && key_18yoResp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_18yoResp.tStart = t;  // (not accounting for frame time here)
      key_18yoResp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_18yoResp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp.clearEvents(); });
    }

    if (key_18yoResp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_18yoResp.getKeys({keyList: [], waitRelease: false});
      _key_18yoResp_allKeys = _key_18yoResp_allKeys.concat(theseKeys);
      if (_key_18yoResp_allKeys.length > 0) {
        key_18yoResp.keys = _key_18yoResp_allKeys[_key_18yoResp_allKeys.length - 1].name;  // just the last key pressed
        key_18yoResp.rt = _key_18yoResp_allKeys[_key_18yoResp_allKeys.length - 1].rt;
        // was this correct?
        if (key_18yoResp.keys == ["y"]) {
            key_18yoResp.corr = 1;
        } else {
            key_18yoResp.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of WelcomeComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function WelcomeRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Welcome' ---
    for (const thisComponent of WelcomeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // was no response the correct answer?!
    if (key_18yoResp.keys === undefined) {
      if (['None','none',undefined].includes(["y"])) {
         key_18yoResp.corr = 1;  // correct non-response
      } else {
         key_18yoResp.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_18yoResp.corr, level);
    }
    psychoJS.experiment.addData('key_18yoResp.keys', key_18yoResp.keys);
    psychoJS.experiment.addData('key_18yoResp.corr', key_18yoResp.corr);
    if (typeof key_18yoResp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_18yoResp.rt', key_18yoResp.rt);
        routineTimer.reset();
        }
    
    key_18yoResp.stop();
    // the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_9_allKeys;
var Survey_prolific_IDComponents;
function Survey_prolific_IDRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_prolific_ID' ---
    t = 0;
    Survey_prolific_IDClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_8.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_8.refresh();
    key_resp_9.keys = undefined;
    key_resp_9.rt = undefined;
    _key_resp_9_allKeys = [];
    // keep track of which components have finished
    Survey_prolific_IDComponents = [];
    Survey_prolific_IDComponents.push(text_31);
    Survey_prolific_IDComponents.push(text_32);
    Survey_prolific_IDComponents.push(textbox_8);
    Survey_prolific_IDComponents.push(key_resp_9);
    
    for (const thisComponent of Survey_prolific_IDComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_prolific_IDRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_prolific_ID' ---
    // get current time
    t = Survey_prolific_IDClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_31* updates
    if (t >= 0.0 && text_31.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_31.tStart = t;  // (not accounting for frame time here)
      text_31.frameNStart = frameN;  // exact frame index
      
      text_31.setAutoDraw(true);
    }

    
    // *text_32* updates
    if (t >= 0.0 && text_32.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_32.tStart = t;  // (not accounting for frame time here)
      text_32.frameNStart = frameN;  // exact frame index
      
      text_32.setAutoDraw(true);
    }

    
    // *textbox_8* updates
    if (t >= 0.0 && textbox_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_8.tStart = t;  // (not accounting for frame time here)
      textbox_8.frameNStart = frameN;  // exact frame index
      
      textbox_8.setAutoDraw(true);
    }

    
    // *key_resp_9* updates
    if (t >= 0.0 && key_resp_9.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_9.tStart = t;  // (not accounting for frame time here)
      key_resp_9.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_9.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_9.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_9.clearEvents(); });
    }

    if (key_resp_9.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_9.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_9_allKeys = _key_resp_9_allKeys.concat(theseKeys);
      if (_key_resp_9_allKeys.length > 0) {
        key_resp_9.keys = _key_resp_9_allKeys[_key_resp_9_allKeys.length - 1].name;  // just the last key pressed
        key_resp_9.rt = _key_resp_9_allKeys[_key_resp_9_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_prolific_IDComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_prolific_IDRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_prolific_ID' ---
    for (const thisComponent of Survey_prolific_IDComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_8.text',textbox_8.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_9.corr, level);
    }
    psychoJS.experiment.addData('key_resp_9.keys', key_resp_9.keys);
    if (typeof key_resp_9.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_9.rt', key_resp_9.rt);
        routineTimer.reset();
        }
    
    key_resp_9.stop();
    // the Routine "Survey_prolific_ID" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_18yoResp_3_allKeys;
var informedconsent_1Components;
function informedconsent_1RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'informedconsent_1' ---
    t = 0;
    informedconsent_1Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_18yoResp_3.keys = undefined;
    key_18yoResp_3.rt = undefined;
    _key_18yoResp_3_allKeys = [];
    // keep track of which components have finished
    informedconsent_1Components = [];
    informedconsent_1Components.push(bakground1_4);
    informedconsent_1Components.push(consent3);
    informedconsent_1Components.push(text_4);
    informedconsent_1Components.push(key_18yoResp_3);
    
    for (const thisComponent of informedconsent_1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function informedconsent_1RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'informedconsent_1' ---
    // get current time
    t = informedconsent_1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *bakground1_4* updates
    if (t >= 0.0 && bakground1_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      bakground1_4.tStart = t;  // (not accounting for frame time here)
      bakground1_4.frameNStart = frameN;  // exact frame index
      
      bakground1_4.setAutoDraw(true);
    }

    
    // *consent3* updates
    if (t >= 0.0 && consent3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      consent3.tStart = t;  // (not accounting for frame time here)
      consent3.frameNStart = frameN;  // exact frame index
      
      consent3.setAutoDraw(true);
    }

    
    // *text_4* updates
    if (t >= 0.0 && text_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_4.tStart = t;  // (not accounting for frame time here)
      text_4.frameNStart = frameN;  // exact frame index
      
      text_4.setAutoDraw(true);
    }

    
    // *key_18yoResp_3* updates
    if (t >= 0.0 && key_18yoResp_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_18yoResp_3.tStart = t;  // (not accounting for frame time here)
      key_18yoResp_3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_18yoResp_3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_3.clearEvents(); });
    }

    if (key_18yoResp_3.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_18yoResp_3.getKeys({keyList: ['a'], waitRelease: false});
      _key_18yoResp_3_allKeys = _key_18yoResp_3_allKeys.concat(theseKeys);
      if (_key_18yoResp_3_allKeys.length > 0) {
        key_18yoResp_3.keys = _key_18yoResp_3_allKeys[_key_18yoResp_3_allKeys.length - 1].name;  // just the last key pressed
        key_18yoResp_3.rt = _key_18yoResp_3_allKeys[_key_18yoResp_3_allKeys.length - 1].rt;
        // was this correct?
        if (key_18yoResp_3.keys == ["y"]) {
            key_18yoResp_3.corr = 1;
        } else {
            key_18yoResp_3.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of informedconsent_1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function informedconsent_1RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'informedconsent_1' ---
    for (const thisComponent of informedconsent_1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // was no response the correct answer?!
    if (key_18yoResp_3.keys === undefined) {
      if (['None','none',undefined].includes(["y"])) {
         key_18yoResp_3.corr = 1;  // correct non-response
      } else {
         key_18yoResp_3.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_18yoResp_3.corr, level);
    }
    psychoJS.experiment.addData('key_18yoResp_3.keys', key_18yoResp_3.keys);
    psychoJS.experiment.addData('key_18yoResp_3.corr', key_18yoResp_3.corr);
    if (typeof key_18yoResp_3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_18yoResp_3.rt', key_18yoResp_3.rt);
        routineTimer.reset();
        }
    
    key_18yoResp_3.stop();
    // the Routine "informedconsent_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_18yoResp_4_allKeys;
var informedconsent_2Components;
function informedconsent_2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'informedconsent_2' ---
    t = 0;
    informedconsent_2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_18yoResp_4.keys = undefined;
    key_18yoResp_4.rt = undefined;
    _key_18yoResp_4_allKeys = [];
    // keep track of which components have finished
    informedconsent_2Components = [];
    informedconsent_2Components.push(bakground1_5);
    informedconsent_2Components.push(consent3_2);
    informedconsent_2Components.push(text_5);
    informedconsent_2Components.push(key_18yoResp_4);
    
    for (const thisComponent of informedconsent_2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function informedconsent_2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'informedconsent_2' ---
    // get current time
    t = informedconsent_2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *bakground1_5* updates
    if (t >= 0.0 && bakground1_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      bakground1_5.tStart = t;  // (not accounting for frame time here)
      bakground1_5.frameNStart = frameN;  // exact frame index
      
      bakground1_5.setAutoDraw(true);
    }

    
    // *consent3_2* updates
    if (t >= 0.0 && consent3_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      consent3_2.tStart = t;  // (not accounting for frame time here)
      consent3_2.frameNStart = frameN;  // exact frame index
      
      consent3_2.setAutoDraw(true);
    }

    
    // *text_5* updates
    if (t >= 0.0 && text_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_5.tStart = t;  // (not accounting for frame time here)
      text_5.frameNStart = frameN;  // exact frame index
      
      text_5.setAutoDraw(true);
    }

    
    // *key_18yoResp_4* updates
    if (t >= 0.0 && key_18yoResp_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_18yoResp_4.tStart = t;  // (not accounting for frame time here)
      key_18yoResp_4.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_18yoResp_4.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_4.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_4.clearEvents(); });
    }

    if (key_18yoResp_4.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_18yoResp_4.getKeys({keyList: ['a'], waitRelease: false});
      _key_18yoResp_4_allKeys = _key_18yoResp_4_allKeys.concat(theseKeys);
      if (_key_18yoResp_4_allKeys.length > 0) {
        key_18yoResp_4.keys = _key_18yoResp_4_allKeys[_key_18yoResp_4_allKeys.length - 1].name;  // just the last key pressed
        key_18yoResp_4.rt = _key_18yoResp_4_allKeys[_key_18yoResp_4_allKeys.length - 1].rt;
        // was this correct?
        if (key_18yoResp_4.keys == ["y"]) {
            key_18yoResp_4.corr = 1;
        } else {
            key_18yoResp_4.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of informedconsent_2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function informedconsent_2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'informedconsent_2' ---
    for (const thisComponent of informedconsent_2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // was no response the correct answer?!
    if (key_18yoResp_4.keys === undefined) {
      if (['None','none',undefined].includes(["y"])) {
         key_18yoResp_4.corr = 1;  // correct non-response
      } else {
         key_18yoResp_4.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_18yoResp_4.corr, level);
    }
    psychoJS.experiment.addData('key_18yoResp_4.keys', key_18yoResp_4.keys);
    psychoJS.experiment.addData('key_18yoResp_4.corr', key_18yoResp_4.corr);
    if (typeof key_18yoResp_4.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_18yoResp_4.rt', key_18yoResp_4.rt);
        routineTimer.reset();
        }
    
    key_18yoResp_4.stop();
    // the Routine "informedconsent_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_18yoResp_5_allKeys;
var informedconsent_3Components;
function informedconsent_3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'informedconsent_3' ---
    t = 0;
    informedconsent_3Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_18yoResp_5.keys = undefined;
    key_18yoResp_5.rt = undefined;
    _key_18yoResp_5_allKeys = [];
    // keep track of which components have finished
    informedconsent_3Components = [];
    informedconsent_3Components.push(bakground1_6);
    informedconsent_3Components.push(consent3_3);
    informedconsent_3Components.push(text_6);
    informedconsent_3Components.push(key_18yoResp_5);
    
    for (const thisComponent of informedconsent_3Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function informedconsent_3RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'informedconsent_3' ---
    // get current time
    t = informedconsent_3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *bakground1_6* updates
    if (t >= 0.0 && bakground1_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      bakground1_6.tStart = t;  // (not accounting for frame time here)
      bakground1_6.frameNStart = frameN;  // exact frame index
      
      bakground1_6.setAutoDraw(true);
    }

    
    // *consent3_3* updates
    if (t >= 0.0 && consent3_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      consent3_3.tStart = t;  // (not accounting for frame time here)
      consent3_3.frameNStart = frameN;  // exact frame index
      
      consent3_3.setAutoDraw(true);
    }

    
    // *text_6* updates
    if (t >= 0.0 && text_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_6.tStart = t;  // (not accounting for frame time here)
      text_6.frameNStart = frameN;  // exact frame index
      
      text_6.setAutoDraw(true);
    }

    
    // *key_18yoResp_5* updates
    if (t >= 0.0 && key_18yoResp_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_18yoResp_5.tStart = t;  // (not accounting for frame time here)
      key_18yoResp_5.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_18yoResp_5.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_5.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_5.clearEvents(); });
    }

    if (key_18yoResp_5.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_18yoResp_5.getKeys({keyList: ['a'], waitRelease: false});
      _key_18yoResp_5_allKeys = _key_18yoResp_5_allKeys.concat(theseKeys);
      if (_key_18yoResp_5_allKeys.length > 0) {
        key_18yoResp_5.keys = _key_18yoResp_5_allKeys[_key_18yoResp_5_allKeys.length - 1].name;  // just the last key pressed
        key_18yoResp_5.rt = _key_18yoResp_5_allKeys[_key_18yoResp_5_allKeys.length - 1].rt;
        // was this correct?
        if (key_18yoResp_5.keys == ["y"]) {
            key_18yoResp_5.corr = 1;
        } else {
            key_18yoResp_5.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of informedconsent_3Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function informedconsent_3RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'informedconsent_3' ---
    for (const thisComponent of informedconsent_3Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // was no response the correct answer?!
    if (key_18yoResp_5.keys === undefined) {
      if (['None','none',undefined].includes(["y"])) {
         key_18yoResp_5.corr = 1;  // correct non-response
      } else {
         key_18yoResp_5.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_18yoResp_5.corr, level);
    }
    psychoJS.experiment.addData('key_18yoResp_5.keys', key_18yoResp_5.keys);
    psychoJS.experiment.addData('key_18yoResp_5.corr', key_18yoResp_5.corr);
    if (typeof key_18yoResp_5.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_18yoResp_5.rt', key_18yoResp_5.rt);
        routineTimer.reset();
        }
    
    key_18yoResp_5.stop();
    // the Routine "informedconsent_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_18yoResp_6_allKeys;
var informedconsent_4Components;
function informedconsent_4RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'informedconsent_4' ---
    t = 0;
    informedconsent_4Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_18yoResp_6.keys = undefined;
    key_18yoResp_6.rt = undefined;
    _key_18yoResp_6_allKeys = [];
    // keep track of which components have finished
    informedconsent_4Components = [];
    informedconsent_4Components.push(bakground1_7);
    informedconsent_4Components.push(consent3_4);
    informedconsent_4Components.push(text_29);
    informedconsent_4Components.push(key_18yoResp_6);
    
    for (const thisComponent of informedconsent_4Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function informedconsent_4RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'informedconsent_4' ---
    // get current time
    t = informedconsent_4Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *bakground1_7* updates
    if (t >= 0.0 && bakground1_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      bakground1_7.tStart = t;  // (not accounting for frame time here)
      bakground1_7.frameNStart = frameN;  // exact frame index
      
      bakground1_7.setAutoDraw(true);
    }

    
    // *consent3_4* updates
    if (t >= 0.0 && consent3_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      consent3_4.tStart = t;  // (not accounting for frame time here)
      consent3_4.frameNStart = frameN;  // exact frame index
      
      consent3_4.setAutoDraw(true);
    }

    
    // *text_29* updates
    if (t >= 0.0 && text_29.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_29.tStart = t;  // (not accounting for frame time here)
      text_29.frameNStart = frameN;  // exact frame index
      
      text_29.setAutoDraw(true);
    }

    
    // *key_18yoResp_6* updates
    if (t >= 0.0 && key_18yoResp_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_18yoResp_6.tStart = t;  // (not accounting for frame time here)
      key_18yoResp_6.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_18yoResp_6.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_6.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_6.clearEvents(); });
    }

    if (key_18yoResp_6.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_18yoResp_6.getKeys({keyList: ['s'], waitRelease: false});
      _key_18yoResp_6_allKeys = _key_18yoResp_6_allKeys.concat(theseKeys);
      if (_key_18yoResp_6_allKeys.length > 0) {
        key_18yoResp_6.keys = _key_18yoResp_6_allKeys[_key_18yoResp_6_allKeys.length - 1].name;  // just the last key pressed
        key_18yoResp_6.rt = _key_18yoResp_6_allKeys[_key_18yoResp_6_allKeys.length - 1].rt;
        // was this correct?
        if (key_18yoResp_6.keys == ["y"]) {
            key_18yoResp_6.corr = 1;
        } else {
            key_18yoResp_6.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of informedconsent_4Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function informedconsent_4RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'informedconsent_4' ---
    for (const thisComponent of informedconsent_4Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // was no response the correct answer?!
    if (key_18yoResp_6.keys === undefined) {
      if (['None','none',undefined].includes(["y"])) {
         key_18yoResp_6.corr = 1;  // correct non-response
      } else {
         key_18yoResp_6.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_18yoResp_6.corr, level);
    }
    psychoJS.experiment.addData('key_18yoResp_6.keys', key_18yoResp_6.keys);
    psychoJS.experiment.addData('key_18yoResp_6.corr', key_18yoResp_6.corr);
    if (typeof key_18yoResp_6.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_18yoResp_6.rt', key_18yoResp_6.rt);
        routineTimer.reset();
        }
    
    key_18yoResp_6.stop();
    // the Routine "informedconsent_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_18yoResp_7_allKeys;
var informedconsent_5Components;
function informedconsent_5RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'informedconsent_5' ---
    t = 0;
    informedconsent_5Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_18yoResp_7.keys = undefined;
    key_18yoResp_7.rt = undefined;
    _key_18yoResp_7_allKeys = [];
    // keep track of which components have finished
    informedconsent_5Components = [];
    informedconsent_5Components.push(bakground1_8);
    informedconsent_5Components.push(consent3_5);
    informedconsent_5Components.push(text_30);
    informedconsent_5Components.push(key_18yoResp_7);
    
    for (const thisComponent of informedconsent_5Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function informedconsent_5RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'informedconsent_5' ---
    // get current time
    t = informedconsent_5Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *bakground1_8* updates
    if (t >= 0.0 && bakground1_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      bakground1_8.tStart = t;  // (not accounting for frame time here)
      bakground1_8.frameNStart = frameN;  // exact frame index
      
      bakground1_8.setAutoDraw(true);
    }

    
    // *consent3_5* updates
    if (t >= 0.0 && consent3_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      consent3_5.tStart = t;  // (not accounting for frame time here)
      consent3_5.frameNStart = frameN;  // exact frame index
      
      consent3_5.setAutoDraw(true);
    }

    
    // *text_30* updates
    if (t >= 0.0 && text_30.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_30.tStart = t;  // (not accounting for frame time here)
      text_30.frameNStart = frameN;  // exact frame index
      
      text_30.setAutoDraw(true);
    }

    
    // *key_18yoResp_7* updates
    if (t >= 0.0 && key_18yoResp_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_18yoResp_7.tStart = t;  // (not accounting for frame time here)
      key_18yoResp_7.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_18yoResp_7.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_7.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_18yoResp_7.clearEvents(); });
    }

    if (key_18yoResp_7.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_18yoResp_7.getKeys({keyList: ['s'], waitRelease: false});
      _key_18yoResp_7_allKeys = _key_18yoResp_7_allKeys.concat(theseKeys);
      if (_key_18yoResp_7_allKeys.length > 0) {
        key_18yoResp_7.keys = _key_18yoResp_7_allKeys[_key_18yoResp_7_allKeys.length - 1].name;  // just the last key pressed
        key_18yoResp_7.rt = _key_18yoResp_7_allKeys[_key_18yoResp_7_allKeys.length - 1].rt;
        // was this correct?
        if (key_18yoResp_7.keys == ["y"]) {
            key_18yoResp_7.corr = 1;
        } else {
            key_18yoResp_7.corr = 0;
        }
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of informedconsent_5Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function informedconsent_5RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'informedconsent_5' ---
    for (const thisComponent of informedconsent_5Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // was no response the correct answer?!
    if (key_18yoResp_7.keys === undefined) {
      if (['None','none',undefined].includes(["y"])) {
         key_18yoResp_7.corr = 1;  // correct non-response
      } else {
         key_18yoResp_7.corr = 0;  // failed to respond (incorrectly)
      }
    }
    // store data for current loop
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_18yoResp_7.corr, level);
    }
    psychoJS.experiment.addData('key_18yoResp_7.keys', key_18yoResp_7.keys);
    psychoJS.experiment.addData('key_18yoResp_7.corr', key_18yoResp_7.corr);
    if (typeof key_18yoResp_7.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_18yoResp_7.rt', key_18yoResp_7.rt);
        routineTimer.reset();
        }
    
    key_18yoResp_7.stop();
    // the Routine "informedconsent_5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _heard_sound_allKeys;
var soundCheckComponents;
function soundCheckRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'soundCheck' ---
    t = 0;
    soundCheckClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    heard_sound.keys = undefined;
    heard_sound.rt = undefined;
    _heard_sound_allKeys = [];
    sound_1.setVolume(1.0);
    // keep track of which components have finished
    soundCheckComponents = [];
    soundCheckComponents.push(heard_sound);
    soundCheckComponents.push(text_2);
    soundCheckComponents.push(mic_check);
    soundCheckComponents.push(sound_1);
    
    for (const thisComponent of soundCheckComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function soundCheckRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'soundCheck' ---
    // get current time
    t = soundCheckClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *heard_sound* updates
    if (t >= 0.0 && heard_sound.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      heard_sound.tStart = t;  // (not accounting for frame time here)
      heard_sound.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { heard_sound.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { heard_sound.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { heard_sound.clearEvents(); });
    }

    if (heard_sound.status === PsychoJS.Status.STARTED) {
      let theseKeys = heard_sound.getKeys({keyList: ['space'], waitRelease: false});
      _heard_sound_allKeys = _heard_sound_allKeys.concat(theseKeys);
      if (_heard_sound_allKeys.length > 0) {
        heard_sound.keys = _heard_sound_allKeys[_heard_sound_allKeys.length - 1].name;  // just the last key pressed
        heard_sound.rt = _heard_sound_allKeys[_heard_sound_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *text_2* updates
    if (t >= 0.0 && text_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_2.tStart = t;  // (not accounting for frame time here)
      text_2.frameNStart = frameN;  // exact frame index
      
      text_2.setAutoDraw(true);
    }

    if (t >= 0.0 && mic_check.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mic_check.tStart = t;  // (not accounting for frame time here)
      mic_check.frameNStart = frameN;  // exact frame index
      
      await mic_check.start();
    }
    frameRemains = 0.0 + 1.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (mic_check.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      mic_check.pause();
    }
    // start/stop sound_1
    if (t >= 0.0 && sound_1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sound_1.tStart = t;  // (not accounting for frame time here)
      sound_1.frameNStart = frameN;  // exact frame index
      
      psychoJS.window.callOnFlip(function(){ sound_1.play(); });  // screen flip
      sound_1.status = PsychoJS.Status.STARTED;
    }
    if (t >= (sound_1.getDuration() + sound_1.tStart)     && sound_1.status === PsychoJS.Status.STARTED) {
      sound_1.stop();  // stop the sound (if longer than duration)
      sound_1.status = PsychoJS.Status.FINISHED;
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of soundCheckComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var thisFilename;
function soundCheckRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'soundCheck' ---
    for (const thisComponent of soundCheckComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    heard_sound.stop();
    // stop the microphone (make the audio data ready for upload)
    await mic_check.stop();
    // construct a filename for this recording
    thisFilename = 'recording_mic_check_' + currentLoop.name + '_' + currentLoop.thisN
    // get the recording
    mic_check.lastClip = await mic_check.getRecording({
      tag: thisFilename + '_' + util.MonotonicClock.getDateStr(),
      flush: false
    });
    psychoJS.experiment.addData('mic_check.clip', thisFilename);
    // start the asynchronous upload to the server
    mic_check.lastClip.upload();
    sound_1.stop();  // ensure sound has stopped at end of routine
    // the Routine "soundCheck" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var verbal_instructionsComponents;
function verbal_instructionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'verbal_instructions' ---
    t = 0;
    verbal_instructionsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    sound_2.setVolume(1.0);
    // keep track of which components have finished
    verbal_instructionsComponents = [];
    verbal_instructionsComponents.push(text_7);
    verbal_instructionsComponents.push(sound_2);
    
    for (const thisComponent of verbal_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function verbal_instructionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'verbal_instructions' ---
    // get current time
    t = verbal_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_7* updates
    if (t >= 0.0 && text_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_7.tStart = t;  // (not accounting for frame time here)
      text_7.frameNStart = frameN;  // exact frame index
      
      text_7.setAutoDraw(true);
    }

    frameRemains = 0.0 + 10.0 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text_7.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_7.setAutoDraw(false);
    }
    // start/stop sound_2
    if (t >= 10.0 && sound_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      sound_2.tStart = t;  // (not accounting for frame time here)
      sound_2.frameNStart = frameN;  // exact frame index
      
      sound_2.play();  // start the sound (it finishes automatically)
      sound_2.status = PsychoJS.Status.STARTED;
    }
    if (t >= (sound_2.getDuration() + sound_2.tStart)     && sound_2.status === PsychoJS.Status.STARTED) {
      sound_2.stop();  // stop the sound (if longer than duration)
      sound_2.status = PsychoJS.Status.FINISHED;
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of verbal_instructionsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function verbal_instructionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'verbal_instructions' ---
    for (const thisComponent of verbal_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    sound_2.stop();  // ensure sound has stopped at end of routine
    // the Routine "verbal_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _continue_space_allKeys;
var task_instructionsComponents;
function task_instructionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'task_instructions' ---
    t = 0;
    task_instructionsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    continue_space.keys = undefined;
    continue_space.rt = undefined;
    _continue_space_allKeys = [];
    // keep track of which components have finished
    task_instructionsComponents = [];
    task_instructionsComponents.push(general_instructions);
    task_instructionsComponents.push(continue_space);
    
    for (const thisComponent of task_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function task_instructionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'task_instructions' ---
    // get current time
    t = task_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *general_instructions* updates
    if (t >= 0.0 && general_instructions.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      general_instructions.tStart = t;  // (not accounting for frame time here)
      general_instructions.frameNStart = frameN;  // exact frame index
      
      general_instructions.setAutoDraw(true);
    }

    
    // *continue_space* updates
    if (t >= 0 && continue_space.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_space.tStart = t;  // (not accounting for frame time here)
      continue_space.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { continue_space.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { continue_space.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { continue_space.clearEvents(); });
    }

    if (continue_space.status === PsychoJS.Status.STARTED) {
      let theseKeys = continue_space.getKeys({keyList: ['s'], waitRelease: false});
      _continue_space_allKeys = _continue_space_allKeys.concat(theseKeys);
      if (_continue_space_allKeys.length > 0) {
        continue_space.keys = _continue_space_allKeys[_continue_space_allKeys.length - 1].name;  // just the last key pressed
        continue_space.rt = _continue_space_allKeys[_continue_space_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of task_instructionsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function task_instructionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'task_instructions' ---
    for (const thisComponent of task_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(continue_space.corr, level);
    }
    psychoJS.experiment.addData('continue_space.keys', continue_space.keys);
    if (typeof continue_space.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('continue_space.rt', continue_space.rt);
        routineTimer.reset();
        }
    
    continue_space.stop();
    // the Routine "task_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'conditions.xlsx',
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial of trials) {
      snapshot = trials.getSnapshot();
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(narrativeExposureRoutineBegin(snapshot));
      trialsLoopScheduler.add(narrativeExposureRoutineEachFrame());
      trialsLoopScheduler.add(narrativeExposureRoutineEnd(snapshot));
      trialsLoopScheduler.add(recall_instructionsRoutineBegin(snapshot));
      trialsLoopScheduler.add(recall_instructionsRoutineEachFrame());
      trialsLoopScheduler.add(recall_instructionsRoutineEnd(snapshot));
      trialsLoopScheduler.add(verbalRecallRoutineBegin(snapshot));
      trialsLoopScheduler.add(verbalRecallRoutineEachFrame());
      trialsLoopScheduler.add(verbalRecallRoutineEnd(snapshot));
      trialsLoopScheduler.add(next_story_instuctRoutineBegin(snapshot));
      trialsLoopScheduler.add(next_story_instuctRoutineEachFrame());
      trialsLoopScheduler.add(next_story_instuctRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var narrativeExposureComponents;
function narrativeExposureRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'narrativeExposure' ---
    t = 0;
    narrativeExposureClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    story_1 = new sound.Sound({
    win: psychoJS.window,
    value: stories,
    secs: -1,
    });
    story_1.setVolume(1.0);
    // keep track of which components have finished
    narrativeExposureComponents = [];
    narrativeExposureComponents.push(text);
    narrativeExposureComponents.push(story_1);
    
    for (const thisComponent of narrativeExposureComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function narrativeExposureRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'narrativeExposure' ---
    // get current time
    t = narrativeExposureClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text* updates
    if (t >= 0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }

    frameRemains = 0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (text.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text.setAutoDraw(false);
    }
    // start/stop story_1
    if (t >= 2 && story_1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      story_1.tStart = t;  // (not accounting for frame time here)
      story_1.frameNStart = frameN;  // exact frame index
      
      story_1.play();  // start the sound (it finishes automatically)
      story_1.status = PsychoJS.Status.STARTED;
    }
    if (t >= (story_1.getDuration() + story_1.tStart)     && story_1.status === PsychoJS.Status.STARTED) {
      story_1.stop();  // stop the sound (if longer than duration)
      story_1.status = PsychoJS.Status.FINISHED;
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of narrativeExposureComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function narrativeExposureRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'narrativeExposure' ---
    for (const thisComponent of narrativeExposureComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    story_1.stop();  // ensure sound has stopped at end of routine
    // the Routine "narrativeExposure" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _continue_space_2_allKeys;
var recall_instructionsComponents;
function recall_instructionsRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'recall_instructions' ---
    t = 0;
    recall_instructionsClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    continue_space_2.keys = undefined;
    continue_space_2.rt = undefined;
    _continue_space_2_allKeys = [];
    // keep track of which components have finished
    recall_instructionsComponents = [];
    recall_instructionsComponents.push(instruction);
    recall_instructionsComponents.push(continue_space_2);
    
    for (const thisComponent of recall_instructionsComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function recall_instructionsRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'recall_instructions' ---
    // get current time
    t = recall_instructionsClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *instruction* updates
    if (t >= 0.0 && instruction.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      instruction.tStart = t;  // (not accounting for frame time here)
      instruction.frameNStart = frameN;  // exact frame index
      
      instruction.setAutoDraw(true);
    }

    
    // *continue_space_2* updates
    if (t >= 0 && continue_space_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_space_2.tStart = t;  // (not accounting for frame time here)
      continue_space_2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { continue_space_2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { continue_space_2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { continue_space_2.clearEvents(); });
    }

    if (continue_space_2.status === PsychoJS.Status.STARTED) {
      let theseKeys = continue_space_2.getKeys({keyList: ['space'], waitRelease: false});
      _continue_space_2_allKeys = _continue_space_2_allKeys.concat(theseKeys);
      if (_continue_space_2_allKeys.length > 0) {
        continue_space_2.keys = _continue_space_2_allKeys[_continue_space_2_allKeys.length - 1].name;  // just the last key pressed
        continue_space_2.rt = _continue_space_2_allKeys[_continue_space_2_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of recall_instructionsComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function recall_instructionsRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'recall_instructions' ---
    for (const thisComponent of recall_instructionsComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(continue_space_2.corr, level);
    }
    psychoJS.experiment.addData('continue_space_2.keys', continue_space_2.keys);
    if (typeof continue_space_2.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('continue_space_2.rt', continue_space_2.rt);
        routineTimer.reset();
        }
    
    continue_space_2.stop();
    // the Routine "recall_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_2_allKeys;
var verbalRecallComponents;
function verbalRecallRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'verbalRecall' ---
    t = 0;
    verbalRecallClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    key_resp_2.keys = undefined;
    key_resp_2.rt = undefined;
    _key_resp_2_allKeys = [];
    // keep track of which components have finished
    verbalRecallComponents = [];
    verbalRecallComponents.push(recording_in_progress);
    verbalRecallComponents.push(recall_instruction);
    verbalRecallComponents.push(polygon);
    verbalRecallComponents.push(key_resp_2);
    verbalRecallComponents.push(polygon_2);
    verbalRecallComponents.push(mic_2);
    
    for (const thisComponent of verbalRecallComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function verbalRecallRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'verbalRecall' ---
    // get current time
    t = verbalRecallClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *recording_in_progress* updates
    if (t >= 0.0 && recording_in_progress.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      recording_in_progress.tStart = t;  // (not accounting for frame time here)
      recording_in_progress.frameNStart = frameN;  // exact frame index
      
      recording_in_progress.setAutoDraw(true);
    }

    
    // *recall_instruction* updates
    if (t >= 0.0 && recall_instruction.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      recall_instruction.tStart = t;  // (not accounting for frame time here)
      recall_instruction.frameNStart = frameN;  // exact frame index
      
      recall_instruction.setAutoDraw(true);
    }

    
    // *polygon* updates
    if (t >= 0.0 && polygon.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon.tStart = t;  // (not accounting for frame time here)
      polygon.frameNStart = frameN;  // exact frame index
      
      polygon.setAutoDraw(true);
    }

    frameRemains = 0.0 + 240 - psychoJS.window.monitorFramePeriod * 0.75;  // most of one frame period left
    if (polygon.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      polygon.setAutoDraw(false);
    }
    
    // *key_resp_2* updates
    if (t >= 240 && key_resp_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_2.tStart = t;  // (not accounting for frame time here)
      key_resp_2.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_2.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_2.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_2.clearEvents(); });
    }

    if (key_resp_2.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_2.getKeys({keyList: ['d'], waitRelease: false});
      _key_resp_2_allKeys = _key_resp_2_allKeys.concat(theseKeys);
      if (_key_resp_2_allKeys.length > 0) {
        key_resp_2.keys = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].name;  // just the last key pressed
        key_resp_2.rt = _key_resp_2_allKeys[_key_resp_2_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *polygon_2* updates
    if (t >= 240 && polygon_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      polygon_2.tStart = t;  // (not accounting for frame time here)
      polygon_2.frameNStart = frameN;  // exact frame index
      
      polygon_2.setAutoDraw(true);
    }

    if (t >= 0.0 && mic_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mic_2.tStart = t;  // (not accounting for frame time here)
      mic_2.frameNStart = frameN;  // exact frame index
      
      await mic_2.start();
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of verbalRecallComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function verbalRecallRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'verbalRecall' ---
    for (const thisComponent of verbalRecallComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_2.corr, level);
    }
    psychoJS.experiment.addData('key_resp_2.keys', key_resp_2.keys);
    if (typeof key_resp_2.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_2.rt', key_resp_2.rt);
        routineTimer.reset();
        }
    
    key_resp_2.stop();
    // stop the microphone (make the audio data ready for upload)
    await mic_2.stop();
    // construct a filename for this recording
    thisFilename = 'recording_mic_2_' + currentLoop.name + '_' + currentLoop.thisN
    // get the recording
    mic_2.lastClip = await mic_2.getRecording({
      tag: thisFilename + '_' + util.MonotonicClock.getDateStr(),
      flush: false
    });
    psychoJS.experiment.addData('mic_2.clip', thisFilename);
    // start the asynchronous upload to the server
    mic_2.lastClip.upload();
    // the Routine "verbalRecall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _continue_space_3_allKeys;
var next_story_instuctComponents;
function next_story_instuctRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'next_story_instuct' ---
    t = 0;
    next_story_instuctClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    continue_space_3.keys = undefined;
    continue_space_3.rt = undefined;
    _continue_space_3_allKeys = [];
    // keep track of which components have finished
    next_story_instuctComponents = [];
    next_story_instuctComponents.push(general_instructions_2);
    next_story_instuctComponents.push(continue_space_3);
    
    for (const thisComponent of next_story_instuctComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function next_story_instuctRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'next_story_instuct' ---
    // get current time
    t = next_story_instuctClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *general_instructions_2* updates
    if (t >= 0.0 && general_instructions_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      general_instructions_2.tStart = t;  // (not accounting for frame time here)
      general_instructions_2.frameNStart = frameN;  // exact frame index
      
      general_instructions_2.setAutoDraw(true);
    }

    
    // *continue_space_3* updates
    if (t >= 0 && continue_space_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      continue_space_3.tStart = t;  // (not accounting for frame time here)
      continue_space_3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { continue_space_3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { continue_space_3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { continue_space_3.clearEvents(); });
    }

    if (continue_space_3.status === PsychoJS.Status.STARTED) {
      let theseKeys = continue_space_3.getKeys({keyList: ['r'], waitRelease: false});
      _continue_space_3_allKeys = _continue_space_3_allKeys.concat(theseKeys);
      if (_continue_space_3_allKeys.length > 0) {
        continue_space_3.keys = _continue_space_3_allKeys[_continue_space_3_allKeys.length - 1].name;  // just the last key pressed
        continue_space_3.rt = _continue_space_3_allKeys[_continue_space_3_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of next_story_instuctComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function next_story_instuctRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'next_story_instuct' ---
    for (const thisComponent of next_story_instuctComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(continue_space_3.corr, level);
    }
    psychoJS.experiment.addData('continue_space_3.keys', continue_space_3.keys);
    if (typeof continue_space_3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('continue_space_3.rt', continue_space_3.rt);
        routineTimer.reset();
        }
    
    continue_space_3.stop();
    // the Routine "next_story_instuct" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var gotValidClick;
var Survey_Likert_Q1Components;
function Survey_Likert_Q1RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_Likert_Q1' ---
    t = 0;
    Survey_Likert_Q1Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    slider_4.reset()
    // setup some python lists for storing info about the mouse
    // current position of the mouse:
    mouse.x = [];
    mouse.y = [];
    mouse.leftButton = [];
    mouse.midButton = [];
    mouse.rightButton = [];
    mouse.time = [];
    mouse.clicked_name = [];
    gotValidClick = false; // until a click is received
    // keep track of which components have finished
    Survey_Likert_Q1Components = [];
    Survey_Likert_Q1Components.push(text_17);
    Survey_Likert_Q1Components.push(text_18);
    Survey_Likert_Q1Components.push(slider_4);
    Survey_Likert_Q1Components.push(image);
    Survey_Likert_Q1Components.push(mouse);
    
    for (const thisComponent of Survey_Likert_Q1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var prevButtonState;
var _mouseButtons;
var _mouseXYs;
function Survey_Likert_Q1RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_Likert_Q1' ---
    // get current time
    t = Survey_Likert_Q1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_17* updates
    if (t >= 0.0 && text_17.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_17.tStart = t;  // (not accounting for frame time here)
      text_17.frameNStart = frameN;  // exact frame index
      
      text_17.setAutoDraw(true);
    }

    
    // *text_18* updates
    if (t >= 0.0 && text_18.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_18.tStart = t;  // (not accounting for frame time here)
      text_18.frameNStart = frameN;  // exact frame index
      
      text_18.setAutoDraw(true);
    }

    
    // *slider_4* updates
    if (t >= 0.0 && slider_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      slider_4.tStart = t;  // (not accounting for frame time here)
      slider_4.frameNStart = frameN;  // exact frame index
      
      slider_4.setAutoDraw(true);
    }

    
    // *image* updates
    if ((slider_4.rating) && image.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image.tStart = t;  // (not accounting for frame time here)
      image.frameNStart = frameN;  // exact frame index
      
      image.setAutoDraw(true);
    }

    // *mouse* updates
    if ((slider_4.rating) && mouse.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse.tStart = t;  // (not accounting for frame time here)
      mouse.frameNStart = frameN;  // exact frame index
      
      mouse.status = PsychoJS.Status.STARTED;
      mouse.mouseClock.reset();
      prevButtonState = mouse.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouse.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          for (const obj of [image]) {
            if (obj.contains(mouse)) {
              gotValidClick = true;
              mouse.clicked_name.push(obj.name)
            }
          }
          _mouseXYs = mouse.getPos();
          mouse.x.push(_mouseXYs[0]);
          mouse.y.push(_mouseXYs[1]);
          mouse.leftButton.push(_mouseButtons[0]);
          mouse.midButton.push(_mouseButtons[1]);
          mouse.rightButton.push(_mouseButtons[2]);
          mouse.time.push(mouse.mouseClock.getTime());
          if (gotValidClick === true) { // abort routine on response
            continueRoutine = false;
          }
        }
      }
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_Likert_Q1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_Likert_Q1RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_Likert_Q1' ---
    for (const thisComponent of Survey_Likert_Q1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('slider_4.response', slider_4.getRating());
    psychoJS.experiment.addData('slider_4.rt', slider_4.getRT());
    // store data for psychoJS.experiment (ExperimentHandler)
    if (mouse.x) {  psychoJS.experiment.addData('mouse.x', mouse.x[0])};
    if (mouse.y) {  psychoJS.experiment.addData('mouse.y', mouse.y[0])};
    if (mouse.leftButton) {  psychoJS.experiment.addData('mouse.leftButton', mouse.leftButton[0])};
    if (mouse.midButton) {  psychoJS.experiment.addData('mouse.midButton', mouse.midButton[0])};
    if (mouse.rightButton) {  psychoJS.experiment.addData('mouse.rightButton', mouse.rightButton[0])};
    if (mouse.time) {  psychoJS.experiment.addData('mouse.time', mouse.time[0])};
    if (mouse.clicked_name) {  psychoJS.experiment.addData('mouse.clicked_name', mouse.clicked_name[0])};
    
    // the Routine "Survey_Likert_Q1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var Survey_Likert_Q2Components;
function Survey_Likert_Q2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_Likert_Q2' ---
    t = 0;
    Survey_Likert_Q2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    slider_2.reset()
    // setup some python lists for storing info about the mouse_2
    // current position of the mouse:
    mouse_2.x = [];
    mouse_2.y = [];
    mouse_2.leftButton = [];
    mouse_2.midButton = [];
    mouse_2.rightButton = [];
    mouse_2.time = [];
    mouse_2.clicked_name = [];
    gotValidClick = false; // until a click is received
    // keep track of which components have finished
    Survey_Likert_Q2Components = [];
    Survey_Likert_Q2Components.push(text_11);
    Survey_Likert_Q2Components.push(text_12);
    Survey_Likert_Q2Components.push(slider_2);
    Survey_Likert_Q2Components.push(image_2);
    Survey_Likert_Q2Components.push(mouse_2);
    
    for (const thisComponent of Survey_Likert_Q2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_Likert_Q2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_Likert_Q2' ---
    // get current time
    t = Survey_Likert_Q2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_11* updates
    if (t >= 0.0 && text_11.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_11.tStart = t;  // (not accounting for frame time here)
      text_11.frameNStart = frameN;  // exact frame index
      
      text_11.setAutoDraw(true);
    }

    
    // *text_12* updates
    if (t >= 0.0 && text_12.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_12.tStart = t;  // (not accounting for frame time here)
      text_12.frameNStart = frameN;  // exact frame index
      
      text_12.setAutoDraw(true);
    }

    
    // *slider_2* updates
    if (t >= 0.0 && slider_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      slider_2.tStart = t;  // (not accounting for frame time here)
      slider_2.frameNStart = frameN;  // exact frame index
      
      slider_2.setAutoDraw(true);
    }

    
    // *image_2* updates
    if ((slider_2.rating) && image_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_2.tStart = t;  // (not accounting for frame time here)
      image_2.frameNStart = frameN;  // exact frame index
      
      image_2.setAutoDraw(true);
    }

    // *mouse_2* updates
    if ((slider_2.rating) && mouse_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_2.tStart = t;  // (not accounting for frame time here)
      mouse_2.frameNStart = frameN;  // exact frame index
      
      mouse_2.status = PsychoJS.Status.STARTED;
      mouse_2.mouseClock.reset();
      prevButtonState = mouse_2.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_2.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouse_2.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          for (const obj of [image]) {
            if (obj.contains(mouse_2)) {
              gotValidClick = true;
              mouse_2.clicked_name.push(obj.name)
            }
          }
          _mouseXYs = mouse_2.getPos();
          mouse_2.x.push(_mouseXYs[0]);
          mouse_2.y.push(_mouseXYs[1]);
          mouse_2.leftButton.push(_mouseButtons[0]);
          mouse_2.midButton.push(_mouseButtons[1]);
          mouse_2.rightButton.push(_mouseButtons[2]);
          mouse_2.time.push(mouse_2.mouseClock.getTime());
          if (gotValidClick === true) { // abort routine on response
            continueRoutine = false;
          }
        }
      }
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_Likert_Q2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_Likert_Q2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_Likert_Q2' ---
    for (const thisComponent of Survey_Likert_Q2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('slider_2.response', slider_2.getRating());
    psychoJS.experiment.addData('slider_2.rt', slider_2.getRT());
    // store data for psychoJS.experiment (ExperimentHandler)
    if (mouse_2.x) {  psychoJS.experiment.addData('mouse_2.x', mouse_2.x[0])};
    if (mouse_2.y) {  psychoJS.experiment.addData('mouse_2.y', mouse_2.y[0])};
    if (mouse_2.leftButton) {  psychoJS.experiment.addData('mouse_2.leftButton', mouse_2.leftButton[0])};
    if (mouse_2.midButton) {  psychoJS.experiment.addData('mouse_2.midButton', mouse_2.midButton[0])};
    if (mouse_2.rightButton) {  psychoJS.experiment.addData('mouse_2.rightButton', mouse_2.rightButton[0])};
    if (mouse_2.time) {  psychoJS.experiment.addData('mouse_2.time', mouse_2.time[0])};
    if (mouse_2.clicked_name) {  psychoJS.experiment.addData('mouse_2.clicked_name', mouse_2.clicked_name[0])};
    
    // the Routine "Survey_Likert_Q2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var Survey_Likert_Q3Components;
function Survey_Likert_Q3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_Likert_Q3' ---
    t = 0;
    Survey_Likert_Q3Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    slider_3.reset()
    // setup some python lists for storing info about the mouse_3
    // current position of the mouse:
    mouse_3.x = [];
    mouse_3.y = [];
    mouse_3.leftButton = [];
    mouse_3.midButton = [];
    mouse_3.rightButton = [];
    mouse_3.time = [];
    mouse_3.clicked_name = [];
    gotValidClick = false; // until a click is received
    // keep track of which components have finished
    Survey_Likert_Q3Components = [];
    Survey_Likert_Q3Components.push(text_13);
    Survey_Likert_Q3Components.push(text_14);
    Survey_Likert_Q3Components.push(slider_3);
    Survey_Likert_Q3Components.push(image_3);
    Survey_Likert_Q3Components.push(mouse_3);
    
    for (const thisComponent of Survey_Likert_Q3Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_Likert_Q3RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_Likert_Q3' ---
    // get current time
    t = Survey_Likert_Q3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_13* updates
    if (t >= 0.0 && text_13.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_13.tStart = t;  // (not accounting for frame time here)
      text_13.frameNStart = frameN;  // exact frame index
      
      text_13.setAutoDraw(true);
    }

    
    // *text_14* updates
    if (t >= 0.0 && text_14.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_14.tStart = t;  // (not accounting for frame time here)
      text_14.frameNStart = frameN;  // exact frame index
      
      text_14.setAutoDraw(true);
    }

    
    // *slider_3* updates
    if (t >= 0.0 && slider_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      slider_3.tStart = t;  // (not accounting for frame time here)
      slider_3.frameNStart = frameN;  // exact frame index
      
      slider_3.setAutoDraw(true);
    }

    
    // *image_3* updates
    if ((slider_3.rating) && image_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image_3.tStart = t;  // (not accounting for frame time here)
      image_3.frameNStart = frameN;  // exact frame index
      
      image_3.setAutoDraw(true);
    }

    // *mouse_3* updates
    if ((slider_3.rating) && mouse_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_3.tStart = t;  // (not accounting for frame time here)
      mouse_3.frameNStart = frameN;  // exact frame index
      
      mouse_3.status = PsychoJS.Status.STARTED;
      mouse_3.mouseClock.reset();
      prevButtonState = mouse_3.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_3.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouse_3.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          for (const obj of [image]) {
            if (obj.contains(mouse_3)) {
              gotValidClick = true;
              mouse_3.clicked_name.push(obj.name)
            }
          }
          _mouseXYs = mouse_3.getPos();
          mouse_3.x.push(_mouseXYs[0]);
          mouse_3.y.push(_mouseXYs[1]);
          mouse_3.leftButton.push(_mouseButtons[0]);
          mouse_3.midButton.push(_mouseButtons[1]);
          mouse_3.rightButton.push(_mouseButtons[2]);
          mouse_3.time.push(mouse_3.mouseClock.getTime());
          if (gotValidClick === true) { // abort routine on response
            continueRoutine = false;
          }
        }
      }
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_Likert_Q3Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_Likert_Q3RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_Likert_Q3' ---
    for (const thisComponent of Survey_Likert_Q3Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('slider_3.response', slider_3.getRating());
    psychoJS.experiment.addData('slider_3.rt', slider_3.getRT());
    // store data for psychoJS.experiment (ExperimentHandler)
    if (mouse_3.x) {  psychoJS.experiment.addData('mouse_3.x', mouse_3.x[0])};
    if (mouse_3.y) {  psychoJS.experiment.addData('mouse_3.y', mouse_3.y[0])};
    if (mouse_3.leftButton) {  psychoJS.experiment.addData('mouse_3.leftButton', mouse_3.leftButton[0])};
    if (mouse_3.midButton) {  psychoJS.experiment.addData('mouse_3.midButton', mouse_3.midButton[0])};
    if (mouse_3.rightButton) {  psychoJS.experiment.addData('mouse_3.rightButton', mouse_3.rightButton[0])};
    if (mouse_3.time) {  psychoJS.experiment.addData('mouse_3.time', mouse_3.time[0])};
    if (mouse_3.clicked_name) {  psychoJS.experiment.addData('mouse_3.clicked_name', mouse_3.clicked_name[0])};
    
    // the Routine "Survey_Likert_Q3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_3_allKeys;
var Survey_OpenEnded_Q1Components;
function Survey_OpenEnded_Q1RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_OpenEnded_Q1' ---
    t = 0;
    Survey_OpenEnded_Q1Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_2.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_2.refresh();
    key_resp_3.keys = undefined;
    key_resp_3.rt = undefined;
    _key_resp_3_allKeys = [];
    // keep track of which components have finished
    Survey_OpenEnded_Q1Components = [];
    Survey_OpenEnded_Q1Components.push(text_15);
    Survey_OpenEnded_Q1Components.push(text_16);
    Survey_OpenEnded_Q1Components.push(textbox_2);
    Survey_OpenEnded_Q1Components.push(key_resp_3);
    
    for (const thisComponent of Survey_OpenEnded_Q1Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_OpenEnded_Q1RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_OpenEnded_Q1' ---
    // get current time
    t = Survey_OpenEnded_Q1Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_15* updates
    if (t >= 0.0 && text_15.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_15.tStart = t;  // (not accounting for frame time here)
      text_15.frameNStart = frameN;  // exact frame index
      
      text_15.setAutoDraw(true);
    }

    
    // *text_16* updates
    if (t >= 0.0 && text_16.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_16.tStart = t;  // (not accounting for frame time here)
      text_16.frameNStart = frameN;  // exact frame index
      
      text_16.setAutoDraw(true);
    }

    
    // *textbox_2* updates
    if (t >= 0.0 && textbox_2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_2.tStart = t;  // (not accounting for frame time here)
      textbox_2.frameNStart = frameN;  // exact frame index
      
      textbox_2.setAutoDraw(true);
    }

    
    // *key_resp_3* updates
    if (t >= 0.0 && key_resp_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_3.tStart = t;  // (not accounting for frame time here)
      key_resp_3.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_3.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_3.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_3.clearEvents(); });
    }

    if (key_resp_3.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_3.getKeys({keyList: ['9'], waitRelease: false});
      _key_resp_3_allKeys = _key_resp_3_allKeys.concat(theseKeys);
      if (_key_resp_3_allKeys.length > 0) {
        key_resp_3.keys = _key_resp_3_allKeys[_key_resp_3_allKeys.length - 1].name;  // just the last key pressed
        key_resp_3.rt = _key_resp_3_allKeys[_key_resp_3_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_OpenEnded_Q1Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_OpenEnded_Q1RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_OpenEnded_Q1' ---
    for (const thisComponent of Survey_OpenEnded_Q1Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_2.text',textbox_2.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_3.corr, level);
    }
    psychoJS.experiment.addData('key_resp_3.keys', key_resp_3.keys);
    if (typeof key_resp_3.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_3.rt', key_resp_3.rt);
        routineTimer.reset();
        }
    
    key_resp_3.stop();
    // the Routine "Survey_OpenEnded_Q1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_4_allKeys;
var Survey_OpenEnded_Q2Components;
function Survey_OpenEnded_Q2RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_OpenEnded_Q2' ---
    t = 0;
    Survey_OpenEnded_Q2Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_3.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_3.refresh();
    key_resp_4.keys = undefined;
    key_resp_4.rt = undefined;
    _key_resp_4_allKeys = [];
    // keep track of which components have finished
    Survey_OpenEnded_Q2Components = [];
    Survey_OpenEnded_Q2Components.push(text_19);
    Survey_OpenEnded_Q2Components.push(text_20);
    Survey_OpenEnded_Q2Components.push(textbox_3);
    Survey_OpenEnded_Q2Components.push(key_resp_4);
    
    for (const thisComponent of Survey_OpenEnded_Q2Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_OpenEnded_Q2RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_OpenEnded_Q2' ---
    // get current time
    t = Survey_OpenEnded_Q2Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_19* updates
    if (t >= 0.0 && text_19.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_19.tStart = t;  // (not accounting for frame time here)
      text_19.frameNStart = frameN;  // exact frame index
      
      text_19.setAutoDraw(true);
    }

    
    // *text_20* updates
    if (t >= 0.0 && text_20.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_20.tStart = t;  // (not accounting for frame time here)
      text_20.frameNStart = frameN;  // exact frame index
      
      text_20.setAutoDraw(true);
    }

    
    // *textbox_3* updates
    if (t >= 0.0 && textbox_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_3.tStart = t;  // (not accounting for frame time here)
      textbox_3.frameNStart = frameN;  // exact frame index
      
      textbox_3.setAutoDraw(true);
    }

    
    // *key_resp_4* updates
    if (t >= 0.0 && key_resp_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_4.tStart = t;  // (not accounting for frame time here)
      key_resp_4.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_4.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_4.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_4.clearEvents(); });
    }

    if (key_resp_4.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_4.getKeys({keyList: ['9'], waitRelease: false});
      _key_resp_4_allKeys = _key_resp_4_allKeys.concat(theseKeys);
      if (_key_resp_4_allKeys.length > 0) {
        key_resp_4.keys = _key_resp_4_allKeys[_key_resp_4_allKeys.length - 1].name;  // just the last key pressed
        key_resp_4.rt = _key_resp_4_allKeys[_key_resp_4_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_OpenEnded_Q2Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_OpenEnded_Q2RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_OpenEnded_Q2' ---
    for (const thisComponent of Survey_OpenEnded_Q2Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_3.text',textbox_3.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_4.corr, level);
    }
    psychoJS.experiment.addData('key_resp_4.keys', key_resp_4.keys);
    if (typeof key_resp_4.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_4.rt', key_resp_4.rt);
        routineTimer.reset();
        }
    
    key_resp_4.stop();
    // the Routine "Survey_OpenEnded_Q2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_5_allKeys;
var Survey_OpenEnded_Q3Components;
function Survey_OpenEnded_Q3RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_OpenEnded_Q3' ---
    t = 0;
    Survey_OpenEnded_Q3Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_4.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_4.refresh();
    key_resp_5.keys = undefined;
    key_resp_5.rt = undefined;
    _key_resp_5_allKeys = [];
    // keep track of which components have finished
    Survey_OpenEnded_Q3Components = [];
    Survey_OpenEnded_Q3Components.push(text_21);
    Survey_OpenEnded_Q3Components.push(text_22);
    Survey_OpenEnded_Q3Components.push(textbox_4);
    Survey_OpenEnded_Q3Components.push(key_resp_5);
    
    for (const thisComponent of Survey_OpenEnded_Q3Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_OpenEnded_Q3RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_OpenEnded_Q3' ---
    // get current time
    t = Survey_OpenEnded_Q3Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_21* updates
    if (t >= 0.0 && text_21.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_21.tStart = t;  // (not accounting for frame time here)
      text_21.frameNStart = frameN;  // exact frame index
      
      text_21.setAutoDraw(true);
    }

    
    // *text_22* updates
    if (t >= 0.0 && text_22.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_22.tStart = t;  // (not accounting for frame time here)
      text_22.frameNStart = frameN;  // exact frame index
      
      text_22.setAutoDraw(true);
    }

    
    // *textbox_4* updates
    if (t >= 0.0 && textbox_4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_4.tStart = t;  // (not accounting for frame time here)
      textbox_4.frameNStart = frameN;  // exact frame index
      
      textbox_4.setAutoDraw(true);
    }

    
    // *key_resp_5* updates
    if (t >= 0.0 && key_resp_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_5.tStart = t;  // (not accounting for frame time here)
      key_resp_5.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_5.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_5.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_5.clearEvents(); });
    }

    if (key_resp_5.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_5.getKeys({keyList: ['9'], waitRelease: false});
      _key_resp_5_allKeys = _key_resp_5_allKeys.concat(theseKeys);
      if (_key_resp_5_allKeys.length > 0) {
        key_resp_5.keys = _key_resp_5_allKeys[_key_resp_5_allKeys.length - 1].name;  // just the last key pressed
        key_resp_5.rt = _key_resp_5_allKeys[_key_resp_5_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_OpenEnded_Q3Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_OpenEnded_Q3RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_OpenEnded_Q3' ---
    for (const thisComponent of Survey_OpenEnded_Q3Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_4.text',textbox_4.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_5.corr, level);
    }
    psychoJS.experiment.addData('key_resp_5.keys', key_resp_5.keys);
    if (typeof key_resp_5.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_5.rt', key_resp_5.rt);
        routineTimer.reset();
        }
    
    key_resp_5.stop();
    // the Routine "Survey_OpenEnded_Q3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_6_allKeys;
var Survey_OpenEnded_Q4Components;
function Survey_OpenEnded_Q4RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_OpenEnded_Q4' ---
    t = 0;
    Survey_OpenEnded_Q4Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_5.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_5.refresh();
    key_resp_6.keys = undefined;
    key_resp_6.rt = undefined;
    _key_resp_6_allKeys = [];
    // keep track of which components have finished
    Survey_OpenEnded_Q4Components = [];
    Survey_OpenEnded_Q4Components.push(text_23);
    Survey_OpenEnded_Q4Components.push(text_24);
    Survey_OpenEnded_Q4Components.push(textbox_5);
    Survey_OpenEnded_Q4Components.push(key_resp_6);
    
    for (const thisComponent of Survey_OpenEnded_Q4Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_OpenEnded_Q4RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_OpenEnded_Q4' ---
    // get current time
    t = Survey_OpenEnded_Q4Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_23* updates
    if (t >= 0.0 && text_23.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_23.tStart = t;  // (not accounting for frame time here)
      text_23.frameNStart = frameN;  // exact frame index
      
      text_23.setAutoDraw(true);
    }

    
    // *text_24* updates
    if (t >= 0.0 && text_24.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_24.tStart = t;  // (not accounting for frame time here)
      text_24.frameNStart = frameN;  // exact frame index
      
      text_24.setAutoDraw(true);
    }

    
    // *textbox_5* updates
    if (t >= 0.0 && textbox_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_5.tStart = t;  // (not accounting for frame time here)
      textbox_5.frameNStart = frameN;  // exact frame index
      
      textbox_5.setAutoDraw(true);
    }

    
    // *key_resp_6* updates
    if (t >= 0.0 && key_resp_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_6.tStart = t;  // (not accounting for frame time here)
      key_resp_6.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_6.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_6.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_6.clearEvents(); });
    }

    if (key_resp_6.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_6.getKeys({keyList: ['9'], waitRelease: false});
      _key_resp_6_allKeys = _key_resp_6_allKeys.concat(theseKeys);
      if (_key_resp_6_allKeys.length > 0) {
        key_resp_6.keys = _key_resp_6_allKeys[_key_resp_6_allKeys.length - 1].name;  // just the last key pressed
        key_resp_6.rt = _key_resp_6_allKeys[_key_resp_6_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_OpenEnded_Q4Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_OpenEnded_Q4RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_OpenEnded_Q4' ---
    for (const thisComponent of Survey_OpenEnded_Q4Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_5.text',textbox_5.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_6.corr, level);
    }
    psychoJS.experiment.addData('key_resp_6.keys', key_resp_6.keys);
    if (typeof key_resp_6.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_6.rt', key_resp_6.rt);
        routineTimer.reset();
        }
    
    key_resp_6.stop();
    // the Routine "Survey_OpenEnded_Q4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_7_allKeys;
var Survey_OpenEnded_Q5Components;
function Survey_OpenEnded_Q5RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_OpenEnded_Q5' ---
    t = 0;
    Survey_OpenEnded_Q5Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_6.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_6.refresh();
    key_resp_7.keys = undefined;
    key_resp_7.rt = undefined;
    _key_resp_7_allKeys = [];
    // keep track of which components have finished
    Survey_OpenEnded_Q5Components = [];
    Survey_OpenEnded_Q5Components.push(text_25);
    Survey_OpenEnded_Q5Components.push(text_26);
    Survey_OpenEnded_Q5Components.push(textbox_6);
    Survey_OpenEnded_Q5Components.push(key_resp_7);
    
    for (const thisComponent of Survey_OpenEnded_Q5Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_OpenEnded_Q5RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_OpenEnded_Q5' ---
    // get current time
    t = Survey_OpenEnded_Q5Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_25* updates
    if (t >= 0.0 && text_25.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_25.tStart = t;  // (not accounting for frame time here)
      text_25.frameNStart = frameN;  // exact frame index
      
      text_25.setAutoDraw(true);
    }

    
    // *text_26* updates
    if (t >= 0.0 && text_26.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_26.tStart = t;  // (not accounting for frame time here)
      text_26.frameNStart = frameN;  // exact frame index
      
      text_26.setAutoDraw(true);
    }

    
    // *textbox_6* updates
    if (t >= 0.0 && textbox_6.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_6.tStart = t;  // (not accounting for frame time here)
      textbox_6.frameNStart = frameN;  // exact frame index
      
      textbox_6.setAutoDraw(true);
    }

    
    // *key_resp_7* updates
    if (t >= 0.0 && key_resp_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_7.tStart = t;  // (not accounting for frame time here)
      key_resp_7.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_7.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_7.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_7.clearEvents(); });
    }

    if (key_resp_7.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_7.getKeys({keyList: ['9'], waitRelease: false});
      _key_resp_7_allKeys = _key_resp_7_allKeys.concat(theseKeys);
      if (_key_resp_7_allKeys.length > 0) {
        key_resp_7.keys = _key_resp_7_allKeys[_key_resp_7_allKeys.length - 1].name;  // just the last key pressed
        key_resp_7.rt = _key_resp_7_allKeys[_key_resp_7_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_OpenEnded_Q5Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_OpenEnded_Q5RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_OpenEnded_Q5' ---
    for (const thisComponent of Survey_OpenEnded_Q5Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_6.text',textbox_6.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_7.corr, level);
    }
    psychoJS.experiment.addData('key_resp_7.keys', key_resp_7.keys);
    if (typeof key_resp_7.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_7.rt', key_resp_7.rt);
        routineTimer.reset();
        }
    
    key_resp_7.stop();
    // the Routine "Survey_OpenEnded_Q5" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var _key_resp_8_allKeys;
var Survey_OpenEnded_Q6Components;
function Survey_OpenEnded_Q6RoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Survey_OpenEnded_Q6' ---
    t = 0;
    Survey_OpenEnded_Q6Clock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    textbox_7.setText('DELETE THIS TEXT AND TYPE HERE');
    textbox_7.refresh();
    key_resp_8.keys = undefined;
    key_resp_8.rt = undefined;
    _key_resp_8_allKeys = [];
    // keep track of which components have finished
    Survey_OpenEnded_Q6Components = [];
    Survey_OpenEnded_Q6Components.push(text_27);
    Survey_OpenEnded_Q6Components.push(text_28);
    Survey_OpenEnded_Q6Components.push(textbox_7);
    Survey_OpenEnded_Q6Components.push(key_resp_8);
    
    for (const thisComponent of Survey_OpenEnded_Q6Components)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function Survey_OpenEnded_Q6RoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Survey_OpenEnded_Q6' ---
    // get current time
    t = Survey_OpenEnded_Q6Clock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_27* updates
    if (t >= 0.0 && text_27.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_27.tStart = t;  // (not accounting for frame time here)
      text_27.frameNStart = frameN;  // exact frame index
      
      text_27.setAutoDraw(true);
    }

    
    // *text_28* updates
    if (t >= 0.0 && text_28.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_28.tStart = t;  // (not accounting for frame time here)
      text_28.frameNStart = frameN;  // exact frame index
      
      text_28.setAutoDraw(true);
    }

    
    // *textbox_7* updates
    if (t >= 0.0 && textbox_7.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      textbox_7.tStart = t;  // (not accounting for frame time here)
      textbox_7.frameNStart = frameN;  // exact frame index
      
      textbox_7.setAutoDraw(true);
    }

    
    // *key_resp_8* updates
    if (t >= 0.0 && key_resp_8.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_8.tStart = t;  // (not accounting for frame time here)
      key_resp_8.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_8.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_8.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_8.clearEvents(); });
    }

    if (key_resp_8.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_8.getKeys({keyList: ['9'], waitRelease: false});
      _key_resp_8_allKeys = _key_resp_8_allKeys.concat(theseKeys);
      if (_key_resp_8_allKeys.length > 0) {
        key_resp_8.keys = _key_resp_8_allKeys[_key_resp_8_allKeys.length - 1].name;  // just the last key pressed
        key_resp_8.rt = _key_resp_8_allKeys[_key_resp_8_allKeys.length - 1].rt;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of Survey_OpenEnded_Q6Components)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function Survey_OpenEnded_Q6RoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Survey_OpenEnded_Q6' ---
    for (const thisComponent of Survey_OpenEnded_Q6Components) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('textbox_7.text',textbox_7.text)
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_8.corr, level);
    }
    psychoJS.experiment.addData('key_resp_8.keys', key_resp_8.keys);
    if (typeof key_resp_8.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_8.rt', key_resp_8.rt);
        routineTimer.reset();
        }
    
    key_resp_8.stop();
    // the Routine "Survey_OpenEnded_Q6" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var debreifComponents;
function debreifRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'debreif' ---
    t = 0;
    debreifClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    // keep track of which components have finished
    debreifComponents = [];
    debreifComponents.push(text_3);
    
    for (const thisComponent of debreifComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function debreifRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'debreif' ---
    // get current time
    t = debreifClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_3* updates
    if (t >= 0.0 && text_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_3.tStart = t;  // (not accounting for frame time here)
      text_3.frameNStart = frameN;  // exact frame index
      
      text_3.setAutoDraw(true);
    }

    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of debreifComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function debreifRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'debreif' ---
    for (const thisComponent of debreifComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    // the Routine "debreif" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
