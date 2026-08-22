import { app } from "/scripts/app.js";

const NODE_NAME = "LoadMiniMaxH3DiffusionModelSGLang";
const HIDDEN_WIDGET_TYPE = "converted-widget:h3-sglang-hybrid-mode";

function isHybridModel(value) {
    const filename = String(value ?? "").split(/[\\/]/).pop().toLowerCase();
    return filename.includes("hybrid");
}

function setWidgetVisible(node, widget, visible) {
    if (!widget) {
        return;
    }

    if (!widget._h3SGLangOriginalProperties) {
        widget._h3SGLangOriginalProperties = {
            type: widget.type,
            computeSize: widget.computeSize,
            draw: widget.draw,
        };
    }

    const original = widget._h3SGLangOriginalProperties;
    const currentlyVisible = !widget.hidden;
    if (currentlyVisible === visible) {
        return;
    }

    const originalSize = original.computeSize?.call(widget);
    const rowHeight = Math.max(0, originalSize?.[1] ?? 20) + 4;
    const heightBeforeToggle = Math.max(node.size[1], node.computeSize()[1]);

    if (visible) {
        widget.hidden = false;
        widget.type = original.type;
        widget.computeSize = original.computeSize;
        widget.draw = original.draw;
        node.setSize([
            node.size[0],
            Math.max(node.computeSize()[1], heightBeforeToggle + rowHeight),
        ]);
    } else {
        widget.hidden = true;
        widget.type = HIDDEN_WIDGET_TYPE;
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        node.setSize([
            node.size[0],
            Math.max(node.computeSize()[1], heightBeforeToggle - rowHeight),
        ]);
    }

    node.graph?.setDirtyCanvas(true, true);
}

function installHybridModeControl(node) {
    if (node._h3SGLangHybridModeInstalled) {
        node._h3SGLangUpdateHybridMode?.();
        return;
    }

    const modelWidget = node.widgets?.find(
        (widget) => widget.name === "model_name",
    );
    const hybridModeWidget = node.widgets?.find(
        (widget) => widget.name === "hybrid_mode",
    );
    if (!modelWidget || !hybridModeWidget) {
        return;
    }

    node._h3SGLangHybridModeInstalled = true;
    const updateHybridModeVisibility = () => {
        setWidgetVisible(
            node,
            hybridModeWidget,
            isHybridModel(modelWidget.value),
        );
    };
    node._h3SGLangUpdateHybridMode = updateHybridModeVisibility;

    const originalCallback = modelWidget.callback;
    modelWidget.callback = function (...args) {
        const callbackResult = originalCallback?.apply(this, args);
        requestAnimationFrame(updateHybridModeVisibility);
        return callbackResult;
    };
    updateHybridModeVisibility();
}

function isTargetNode(node) {
    return node.comfyClass === NODE_NAME || node.type === NODE_NAME;
}

app.registerExtension({
    name: "H3.SGLang.HybridMode",
    nodeCreated(node) {
        if (isTargetNode(node)) {
            requestAnimationFrame(() => installHybridModeControl(node));
        }
    },
    loadedGraphNode(node) {
        if (isTargetNode(node)) {
            requestAnimationFrame(() => installHybridModeControl(node));
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            requestAnimationFrame(() => installHybridModeControl(this));
            return result;
        };
    },
});
