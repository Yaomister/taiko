import {
  BPMChangeCommand,
  CommandType,
  Course,
  DelayCommand,
  MeasureCommand,
  MeasureValue,
  Note,
  NoteSequence,
  Song,
} from "tja";

// Use an enum for NoteType instead of union of strings
export enum NoteType {
  Don = "don",
  Ka = "ka",
  BigDon = "bigDon",
  BigKa = "bigKa",
  Drumroll = "drumroll",
  BigDrumroll = "bigDrumroll",
  Balloon = "balloon",
}

export type CourseNote = { timeMs: number; type: NoteType };

// Use an enum for ParseState
export enum ParseState {
  Neutral = "neutral",
  BranchFinding = "branchfinding",
  EndFinding = "endfinding",
  NoteParsing = "noteparsing",
}

type HeldState =
  | { type: NoteType.Drumroll; startTime: number; isBig: boolean }
  | { type: NoteType.Balloon; startTime: number };

// Helpers
function noteType(note: Note): NoteType | null {
  if (note.isDon) return NoteType.Don;
  if (note.isKa) return NoteType.Ka;
  if (note.isBigDon) return NoteType.BigDon;
  if (note.isBigKa) return NoteType.BigKa;
  if (note.isDrumroll) return NoteType.Drumroll;
  if (note.isBigDrumroll) return NoteType.BigDrumroll;
  if (note.isBalloon) return NoteType.Balloon;
  return null;
}

function handleBarLineOffContext(context: HandlerContext) {
  if (
    context.parseState === ParseState.BranchFinding ||
    context.parseState === ParseState.EndFinding
  ) {
    context.i += 1;
    return;
  }
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handleBarLineOnContext(context: HandlerContext) {
  if (
    context.parseState === ParseState.BranchFinding ||
    context.parseState === ParseState.EndFinding
  ) {
    context.i += 1;
    return;
  }
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handleBPMChangeContext(
  context: HandlerContext,
  cmd: BPMChangeCommand,
) {
  if (
    context.parseState === ParseState.BranchFinding ||
    context.parseState === ParseState.EndFinding
  ) {
    context.i += 1;
    return;
  }
  context.bpm = cmd.value;
  const measureLength = (240000 * context.timeSignature.fraction) / context.bpm;
  const top = context.timingStack[context.timingStack.length - 1];
  if (top.time === context.time) {
    top.measureLength = measureLength;
  } else {
    context.timingStack.push({ time: context.time, measureLength });
  }
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handleBranchEndContext(context: HandlerContext) {
  if (context.parseState !== ParseState.BranchFinding) {
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
    return;
  }
  context.parseState = ParseState.Neutral;
  context.i = context.lastHandledCommandIndex;
}

function handleBranchStartContext(context: HandlerContext) {
  context.parseState = ParseState.BranchFinding;
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handleBranchMarkerContext(context: HandlerContext) {
  context.lastHandledCommandIndex = context.i;
  context.i += 1;
}

function handleDelayContext(context: HandlerContext, cmd: DelayCommand) {
  if (
    context.parseState === ParseState.BranchFinding ||
    context.parseState === ParseState.EndFinding
  ) {
    context.i += 1;
    return;
  }
  context.time += cmd.value * 1000;
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handleMeasureContext(context: HandlerContext, cmd: MeasureCommand) {
  if (
    context.parseState === ParseState.BranchFinding ||
    context.parseState === ParseState.EndFinding
  ) {
    context.i += 1;
    return;
  }
  context.timeSignature = cmd.value;
  const measureLength = (240000 * context.timeSignature.fraction) / context.bpm;
  const top = context.timingStack[context.timingStack.length - 1];
  if (top.time === context.time) {
    top.measureLength = measureLength;
  } else {
    context.timingStack.push({ time: context.time, measureLength });
  }
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handlePlayableNoteContext(context: HandlerContext, note: Note) {
  if (note.isDon || note.isKa || note.isBigDon || note.isBigKa) {
    context.notes.push({ timeMs: context.time, type: noteType(note)! });
    return;
  }
  if (note.isBalloon) {
    context.heldState = { type: NoteType.Balloon, startTime: context.time };
    return;
  }
  if (note.isDrumroll) {
    context.heldState = {
      type: NoteType.Drumroll,
      startTime: context.time,
      isBig: false,
    };
    return;
  }
  if (note.isBigDrumroll) {
    context.heldState = {
      type: NoteType.Drumroll,
      startTime: context.time,
      isBig: true,
    };
    return;
  }
  if (note.isEndOfDrumroll && context.heldState) {
    if (context.heldState.type === NoteType.Drumroll) {
      context.notes.push({
        timeMs: context.heldState.startTime,
        type: context.heldState.isBig
          ? NoteType.BigDrumroll
          : NoteType.Drumroll,
      });
    } else {
      context.notes.push({
        timeMs: context.heldState.startTime,
        type: NoteType.Balloon,
      });
    }
    context.balloonIndex += context.heldState.type === NoteType.Balloon ? 1 : 0;
    context.heldState = null;
  }
}

function handleNoteSequenceContext(context: HandlerContext, cmd: NoteSequence) {
  if (context.parseState === ParseState.BranchFinding) {
    context.i += 1;
    return;
  }
  const lastNote = cmd.notes[cmd.notes.length - 1];
  if (context.parseState !== ParseState.NoteParsing) {
    if (lastNote?.isMeasureEnd) {
      context.sequenceNoteCount += cmd.notes.length - 1;
      context.parseState = ParseState.NoteParsing;
      if (context.i !== context.lastHandledCommandIndex) {
        context.i = context.lastHandledCommandIndex;
        return;
      }
    } else {
      context.sequenceNoteCount += cmd.notes.length;
      context.parseState = ParseState.EndFinding;
      context.i += 1;
      return;
    }
  }

  const measureLength =
    context.timingStack[context.timingStack.length - 1].measureLength;
  const dt = measureLength / Math.max(1, context.sequenceNoteCount);

  for (const note of cmd.notes) {
    if (note.isMeasureEnd) {
      if (cmd.notes.length === 1) context.time += dt;
      continue;
    }
    if (note.isBlank) {
      context.time += dt;
      continue;
    }
    handlePlayableNoteContext(context, note);
    context.time += dt;
  }

  if (lastNote?.isMeasureEnd) {
    context.sequenceNoteCount = 0;
    context.parseState = ParseState.Neutral;
  }
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

function handleScrollContext(context: HandlerContext) {
  if (
    context.parseState === ParseState.BranchFinding ||
    context.parseState === ParseState.EndFinding
  ) {
    context.i += 1;
    return;
  }
  context.i += 1;
  context.lastHandledCommandIndex = context.i;
}

// HandlerContext type
type HandlerContext = {
  notes: CourseNote[];
  bpm: number;
  time: number;
  timeSignature: MeasureValue;
  i: number;
  lastHandledCommandIndex: number;
  sequenceNoteCount: number;
  parseState: ParseState;
  heldState: HeldState | null;
  balloonIndex: number;
  timingStack: { time: number; measureLength: number }[];
};

// Main function
export function getCourseNoteTimes(song: Song, course: Course): CourseNote[] {
  const notes: CourseNote[] = [];
  const commands = course.singleCourse.getCommands();

  const context: HandlerContext = {
    notes,
    bpm: song.bpm,
    time: -song.offset * 1000,
    timeSignature: new MeasureValue(4, 4),
    i: 0,
    lastHandledCommandIndex: 0,
    sequenceNoteCount: 0,
    parseState: ParseState.Neutral,
    heldState: null,
    balloonIndex: 0,
    timingStack: [],
  };

  const beatLength0 = 60000 / song.bpm;
  context.timingStack.push({
    time: context.time,
    measureLength: beatLength0 * 4,
  });

  while (context.i < commands.length) {
    const cmd = commands[context.i];
    switch (cmd.commandType) {
      case CommandType.BarLineOff:
        handleBarLineOffContext(context);
        break;
      case CommandType.BarLineOn:
        handleBarLineOnContext(context);
        break;
      case CommandType.BPMChange:
        handleBPMChangeContext(context, cmd as BPMChangeCommand);
        break;
      case CommandType.BranchEnd:
      case CommandType.Section:
        handleBranchEndContext(context);
        break;
      case CommandType.BranchMarker:
        handleBranchMarkerContext(context);
        break;
      case CommandType.BranchStart:
        handleBranchStartContext(context);
        break;
      case CommandType.Delay:
        handleDelayContext(context, cmd as DelayCommand);
        break;
      case CommandType.GoGoEnd:
      case CommandType.GoGoStart:
        context.i += 1;
        context.lastHandledCommandIndex = context.i;
        break;
      case CommandType.Measure:
        handleMeasureContext(context, cmd as MeasureCommand);
        break;
      case CommandType.NoteSequence:
        handleNoteSequenceContext(context, cmd as NoteSequence);
        break;
      case CommandType.Scroll:
        handleScrollContext(context);
        break;
      default: {
        context.i += 1;
        const inBranch =
          context.parseState === ParseState.BranchFinding ||
          context.parseState === ParseState.EndFinding;
        if (!inBranch) context.lastHandledCommandIndex = context.i;
      }
    }
  }

  return context.notes;
}
