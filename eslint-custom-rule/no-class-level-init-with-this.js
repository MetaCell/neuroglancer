export default {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow `this` usage in class field initializers for dependent instance creation",
      recommended: false,
    },
    messages: {
      noThisReference:
        "Do not reference `this` in class field initializers when creating a new instance. Move it to the constructor instead.",
    },
    schema: [],
  },

  create(context) {
    function isInArrowFunction(node) {
      if (!node) {
        return false;
      }
      if (
        node.type === "ArrowFunctionExpression" ||
        node.type === "FunctionExpression"
      ) {
        return true;
      }
      return isInArrowFunction(node.parent);
    }

    function callExpressionNumber(node, found = 0) {
      if (!node) {
        return found;
      }
      return callExpressionNumber(
        node.parent,
        found + (node.type === "CallExpression" ? 1 : 0),
      );
    }

    function isRootCall(node) {
      return callExpressionNumber(node) <= 1;
    }

    function containsThisExpression(node) {
      if (!node || typeof node !== "object") return false;
      if (
        node.type === "ArrowFunctionExpression" ||
        node.type === "FunctionExpression"
      ) {
        return false;
      }
      if (node.type === "ThisExpression") {
        return !isInArrowFunction(node) && !isRootCall(node);
      }

      if (node.type === "CallExpression") {
        // If we have a call, we check the presence of "this" in the arguments
        // console.log("Checking", node)

        for (const argument of node.arguments) {
          if (containsThisExpression(argument)) return true;
        }
        // We check if the callee is not a lambda that is directly called
        const callee = node.callee;
        if (
          callee.type === "ArrowFunctionExpression" &&
          containsThisExpression(callee.body)
        ) {
          return true;
        }
        if (containsThisExpression(callee)) {
          return true;
        }
        // We visited the arguments and the callee, no need to go further
        return false;
      }

      // We walk down the AST for other nodes
      for (const key of Object.keys(node)) {
        if (key === "parent") {
          return false;
        }
        const value = node[key];

        if (Array.isArray(value)) {
          for (const el of value) {
            if (containsThisExpression(el)) return true;
          }
        } else if (value && typeof value === "object") {
          if (containsThisExpression(value)) return true;
        }
      }

      return false;
    }

    return {
      // For class fields (in modern JS/TS)
      PropertyDefinition(node) {
        if (!node.value) return;
        if (containsThisExpression(node.value)) {
          context.report({
            node,
            messageId: "noThisReference",
          });
        }
      },
    };
  },
};
