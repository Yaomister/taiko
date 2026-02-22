import { CommandType, MeasureValue, } from "tja";
// Use an enum for NoteType instead of union of strings
export var NoteType;
(function (NoteType) {
    NoteType["Don"] = "don";
    NoteType["Ka"] = "ka";
    NoteType["BigDon"] = "bigDon";
    NoteType["BigKa"] = "bigKa";
    NoteType["Drumroll"] = "drumroll";
    NoteType["BigDrumroll"] = "bigDrumroll";
    NoteType["Balloon"] = "balloon";
})(NoteType || (NoteType = {}));
// Use an enum for ParseState
export var ParseState;
(function (ParseState) {
    ParseState["Neutral"] = "neutral";
    ParseState["BranchFinding"] = "branchfinding";
    ParseState["EndFinding"] = "endfinding";
    ParseState["NoteParsing"] = "noteparsing";
})(ParseState || (ParseState = {}));
// Helpers
function noteType(note) {
    if (note.isDon)
        return NoteType.Don;
    if (note.isKa)
        return NoteType.Ka;
    if (note.isBigDon)
        return NoteType.BigDon;
    if (note.isBigKa)
        return NoteType.BigKa;
    if (note.isDrumroll)
        return NoteType.Drumroll;
    if (note.isBigDrumroll)
        return NoteType.BigDrumroll;
    if (note.isBalloon)
        return NoteType.Balloon;
    return null;
}
function handleBarLineOffContext(context) {
    if (context.parseState === ParseState.BranchFinding ||
        context.parseState === ParseState.EndFinding) {
        context.i += 1;
        return;
    }
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
function handleBarLineOnContext(context) {
    if (context.parseState === ParseState.BranchFinding ||
        context.parseState === ParseState.EndFinding) {
        context.i += 1;
        return;
    }
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
function handleBPMChangeContext(context, cmd) {
    if (context.parseState === ParseState.BranchFinding ||
        context.parseState === ParseState.EndFinding) {
        context.i += 1;
        return;
    }
    context.bpm = cmd.value;
    const measureLength = (240000 * context.timeSignature.fraction) / context.bpm;
    const top = context.timingStack[context.timingStack.length - 1];
    if (top.time === context.time) {
        top.measureLength = measureLength;
    }
    else {
        context.timingStack.push({ time: context.time, measureLength });
    }
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
function handleBranchEndContext(context) {
    if (context.parseState !== ParseState.BranchFinding) {
        context.i += 1;
        context.lastHandledCommandIndex = context.i;
        return;
    }
    context.parseState = ParseState.Neutral;
    context.i = context.lastHandledCommandIndex;
}
function handleBranchStartContext(context) {
    context.parseState = ParseState.BranchFinding;
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
function handleBranchMarkerContext(context) {
    context.lastHandledCommandIndex = context.i;
    context.i += 1;
}
function handleDelayContext(context, cmd) {
    if (context.parseState === ParseState.BranchFinding ||
        context.parseState === ParseState.EndFinding) {
        context.i += 1;
        return;
    }
    context.time += cmd.value * 1000;
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
function handleMeasureContext(context, cmd) {
    if (context.parseState === ParseState.BranchFinding ||
        context.parseState === ParseState.EndFinding) {
        context.i += 1;
        return;
    }
    context.timeSignature = cmd.value;
    const measureLength = (240000 * context.timeSignature.fraction) / context.bpm;
    const top = context.timingStack[context.timingStack.length - 1];
    if (top.time === context.time) {
        top.measureLength = measureLength;
    }
    else {
        context.timingStack.push({ time: context.time, measureLength });
    }
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
function handlePlayableNoteContext(context, note) {
    if (note.isDon || note.isKa || note.isBigDon || note.isBigKa) {
        context.notes.push({ timeMs: context.time, type: noteType(note) });
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
        }
        else {
            context.notes.push({
                timeMs: context.heldState.startTime,
                type: NoteType.Balloon,
            });
        }
        context.balloonIndex += context.heldState.type === NoteType.Balloon ? 1 : 0;
        context.heldState = null;
    }
}
function handleNoteSequenceContext(context, cmd) {
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
        }
        else {
            context.sequenceNoteCount += cmd.notes.length;
            context.parseState = ParseState.EndFinding;
            context.i += 1;
            return;
        }
    }
    const measureLength = context.timingStack[context.timingStack.length - 1].measureLength;
    const dt = measureLength / Math.max(1, context.sequenceNoteCount);
    for (const note of cmd.notes) {
        if (note.isMeasureEnd) {
            if (cmd.notes.length === 1)
                context.time += dt;
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
function handleScrollContext(context) {
    if (context.parseState === ParseState.BranchFinding ||
        context.parseState === ParseState.EndFinding) {
        context.i += 1;
        return;
    }
    context.i += 1;
    context.lastHandledCommandIndex = context.i;
}
// Main function
export function getCourseNoteTimes(song, course) {
    const notes = [];
    const commands = course.singleCourse.getCommands();
    const context = {
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
                handleBPMChangeContext(context, cmd);
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
                handleDelayContext(context, cmd);
                break;
            case CommandType.GoGoEnd:
            case CommandType.GoGoStart:
                context.i += 1;
                context.lastHandledCommandIndex = context.i;
                break;
            case CommandType.Measure:
                handleMeasureContext(context, cmd);
                break;
            case CommandType.NoteSequence:
                handleNoteSequenceContext(context, cmd);
                break;
            case CommandType.Scroll:
                handleScrollContext(context);
                break;
            default: {
                context.i += 1;
                const inBranch = context.parseState === ParseState.BranchFinding ||
                    context.parseState === ParseState.EndFinding;
                if (!inBranch)
                    context.lastHandledCommandIndex = context.i;
            }
        }
    }
    return context.notes;
}
